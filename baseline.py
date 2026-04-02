# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import sys
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Try to import SuperPoint and SuperGlue (optional - will fallback to CosPlace-only if not available)
try:
    # Try common SuperGlue installation paths
    superglue_paths = [
        "SuperGluePretrainedNetwork/models",
        "../SuperGluePretrainedNetwork/models",
        os.path.join(os.path.dirname(__file__), "..", "SuperGluePretrainedNetwork", "models"),
    ]
    
    superglue_available = False
    for path in superglue_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            try:
                from superpoint import SuperPoint
                from superglue import SuperGlue
                superglue_available = True
                logging.info(f"SuperGlue models found at: {abs_path}")
                break
            except ImportError:
                continue
    
    if not superglue_available:
        logging.warning("SuperGlue models not found. Will use CosPlace-only matching.")
        SuperPoint = None
        SuperGlue = None
except Exception as e:
    logging.warning(f"Could not import SuperGlue: {e}. Will use CosPlace-only matching.")
    SuperPoint = None
    SuperGlue = None


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # NOTE: Original implementation used SIFT+VLAD+KMeans. We now switch to
        # CosPlace descriptors (as in cosplace_superglue_(3)(1).py) for image
        # retrieval while keeping the navigation/exploration game logic intact.

        # --- CosPlace / descriptor-related state ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cosplace_model = None
        self.cosplace_transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Database of global image descriptors (one per exploration image) and
        # nearest-neighbor index
        self.database = None
        self.tree = None
        self.goal = None
        # Store image filenames (without extension) to maintain order and enable flexible loading
        self.image_filenames = None
        
        # --- SuperGlue / geometric verification state ---
        self.superpoint = None
        self.superglue = None
        self.superglue_available = SuperPoint is not None and SuperGlue is not None
        
        # Camera intrinsics (default values - may need adjustment for your setup)
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.K = np.array([[92.,   0., 160.],
                          [ 0.,  92., 120.],
                          [ 0.,   0.,  1. ]], dtype=np.float64)
        
        # SuperGlue verification parameters
        self.cosplace_top_k = 5  # Number of CosPlace candidates to verify with SuperGlue
        self.min_inliers = 30    # Minimum inliers for a valid match

    # ------------------------------------------------------------------
    # SuperGlue helpers (adapted from cosplace_superglue_(3)(1).py)
    # ------------------------------------------------------------------
    def load_superglue_models(self):
        """
        Initialize SuperPoint and SuperGlue with typical indoor settings.
        Returns (superpoint, superglue) or (None, None) if not available.
        """
        if not self.superglue_available:
            logging.warning("SuperGlue models not available. Skipping initialization.")
            return None, None
        
        try:
            superpoint_config = {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024,
            }

            superglue_config = {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }

            sp = SuperPoint(superpoint_config).eval().to(self.device)
            sg = SuperGlue(superglue_config).eval().to(self.device)
            
            logging.info("SuperPoint and SuperGlue models loaded successfully.")
            return sp, sg
        except Exception as e:
            logging.error(f"Failed to load SuperGlue models: {e}")
            return None, None
    
    def _ensure_superglue_models(self):
        """Lazy-initialize SuperGlue models if needed."""
        if self.superglue_available and self.superpoint is None:
            self.superpoint, self.superglue = self.load_superglue_models()
    
    def match_superglue(self, img1_gray, img2_gray):
        """
        Run SuperPoint + SuperGlue between two grayscale images.
        Returns two arrays of matched 2D points (pts1, pts2) or (None, None).
        """
        if not self.superglue_available or self.superpoint is None or self.superglue is None:
            return None, None
        
        t1 = torch.from_numpy(img1_gray).float() / 255.0
        t2 = torch.from_numpy(img2_gray).float() / 255.0

        t1 = t1[None, None].to(self.device)   # (1,1,H,W)
        t2 = t2[None, None].to(self.device)   # (1,1,H,W)

        with torch.no_grad():
            sp0 = self.superpoint({'image': t1})
            sp1 = self.superpoint({'image': t2})

            kpts0 = sp0['keypoints'][0]
            kpts1 = sp1['keypoints'][0]
            scores0 = sp0['scores'][0]
            scores1 = sp1['scores'][0]
            desc0 = sp0['descriptors'][0]
            desc1 = sp1['descriptors'][0]

            data = {
                'image0': t1,
                'image1': t2,
                'keypoints0': kpts0[None],
                'keypoints1': kpts1[None],
                'scores0': scores0[None],
                'scores1': scores1[None],
                'descriptors0': desc0[None],
                'descriptors1': desc1[None],
            }

            pred = self.superglue(data)

            matches0 = pred['matches0'][0].cpu().numpy()
            kpts0_np = kpts0.cpu().numpy()
            kpts1_np = kpts1.cpu().numpy()

        valid = matches0 > -1
        if valid.sum() < 8:
            return None, None

        mkpts0 = kpts0_np[valid]
        mkpts1 = kpts1_np[matches0[valid]]

        pts1 = mkpts0.astype(np.float32)
        pts2 = mkpts1.astype(np.float32)

        return pts1, pts2
    
    def estimate_motion(self, pts1, pts2):
        """
        Estimate relative motion between two frames using Essential matrix + recoverPose.
        Returns (R, t, mask_pose) or (None, None, None) if estimation fails.
        """
        if pts1 is None or pts2 is None or len(pts1) < 8:
            return None, None, None

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None:
            return None, None, None

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask_pose
    
    def verify_match_with_superglue(self, img1_bgr, img2_bgr):
        """
        Verify a match between two images using SuperGlue.
        Returns (num_inliers, is_valid) where is_valid is True if inliers >= min_inliers.
        """
        if not self.superglue_available:
            return 0, False
        
        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY) if len(img1_bgr.shape) == 3 else img1_bgr
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY) if len(img2_bgr.shape) == 3 else img2_bgr
        
        pts1, pts2 = self.match_superglue(img1_gray, img2_gray)
        if pts1 is None or pts2 is None:
            return 0, False
        
        R, t, mask_pose = self.estimate_motion(pts1, pts2)
        if mask_pose is None:
            return 0, False
        
        num_inliers = int(mask_pose.sum())
        is_valid = num_inliers >= self.min_inliers
        
        return num_inliers, is_valid

    # ------------------------------------------------------------------
    # CosPlace helpers (adapted from cosplace_superglue_(3)(1).py)
    # ------------------------------------------------------------------
    def load_cosplace_model(self):
        """
        Load a pretrained CosPlace model from torch.hub.
        This mirrors the usage in cosplace_superglue_(3)(1).py.
        """
        try:
            logging.info("Loading CosPlace model from torch.hub (gmberton/cosplace)...")
            model = torch.hub.load(
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet18",
                fc_output_dim=512,
                trust_repo=True,
            )
            model = model.eval().to(self.device)
            logging.info(f"CosPlace model loaded on device: {self.device}")
            return model
        except Exception as e:
            logging.error(f"Failed to load CosPlace model: {e}")
            return None

    def _ensure_cosplace_model(self):
        """Lazy-initialize the CosPlace model if needed."""
        if self.cosplace_model is None:
            self.cosplace_model = self.load_cosplace_model()

    def compute_cosplace_descriptor_from_img(self, img_bgr):
        """
        Compute a single L2-normalized CosPlace descriptor for a BGR OpenCV image.
        Returns a (1, D) numpy array or None if model is unavailable.
        """
        self._ensure_cosplace_model()
        if self.cosplace_model is None:
            logging.error("CosPlace model is not available; cannot compute descriptor.")
            return None

        # Convert BGR (OpenCV) -> RGB (PIL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGB")

        try:
            t = self.cosplace_transform(pil_img)  # (C,H,W)
        except Exception as e:
            logging.error(f"Failed to transform image for CosPlace: {e}")
            return None

        t = t.unsqueeze(0).to(self.device)  # (1,C,H,W)

        with torch.no_grad():
            feats = self.cosplace_model(t)        # (1,D)
            feats = F.normalize(feats, p=2, dim=1)

        return feats.cpu().numpy().astype(np.float32)  # (1,D)


    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV.
        Supports both .jpg and .png extensions, trying both if needed.
        """
        if self.image_filenames is not None and id < len(self.image_filenames):
            # Use the stored filename (with extension) if available
            img_name = self.image_filenames[id]
            path = os.path.join(self.save_dir, img_name)
        else:
            # Fallback: try both extensions
            path_jpg = os.path.join(self.save_dir, f"{id}.jpg")
            path_png = os.path.join(self.save_dir, f"{id}.png")
            if os.path.exists(path_jpg):
                path = path_jpg
            elif os.path.exists(path_png):
                path = path_png
            else:
                print(f"Image with ID {id} does not exist (tried both .jpg and .png)")
                return
        
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                cv2.imshow(window_name, img)
                cv2.waitKey(1)
            else:
                print(f"Failed to read image: {path}")
        else:
            print(f"Image with ID {id} does not exist at: {path}")

    def get_neighbor(self, img):
        """
        Find the nearest neighbor using CosPlace for retrieval + SuperGlue for verification.
        Strategy: CosPlace for recall (top-K candidates) -> SuperGlue for precision (best match).
        """
        # Compute CosPlace descriptor for the query image
        q_desc = self.compute_cosplace_descriptor_from_img(img)
        if q_desc is None or self.tree is None:
            logging.warning("CosPlace descriptor or BallTree not available; returning index 0.")
            return 0

        # Step 1: Get top-K candidates from CosPlace
        k = self.cosplace_top_k if self.superglue_available else 1
        k = min(k, len(self.database)) if self.database is not None else 1
        
        distances, candidate_indices = self.tree.query(q_desc, k=k)
        candidate_indices = candidate_indices[0]  # Shape: (k,)
        distances = distances[0]  # Shape: (k,)
        
        # If SuperGlue is not available, just return the top CosPlace match
        if not self.superglue_available or self.superpoint is None:
            return int(candidate_indices[0])
        
        # Step 2: Verify each candidate with SuperGlue and pick the best
        self._ensure_superglue_models()
        if self.superpoint is None:
            # Fallback to CosPlace-only
            return int(candidate_indices[0])
        
        best_index = int(candidate_indices[0])
        best_inliers = 0
        
        # Load candidate images and verify with SuperGlue
        for idx in candidate_indices:
            idx = int(idx)
            if idx >= len(self.image_filenames):
                continue
            
            # Load candidate image from database
            candidate_path = os.path.join(self.save_dir, self.image_filenames[idx])
            candidate_img = cv2.imread(candidate_path)
            if candidate_img is None:
                continue
            
            # Verify match with SuperGlue
            num_inliers, is_valid = self.verify_match_with_superglue(img, candidate_img)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_index = idx
            
            # If we found a good match (above threshold), use it
            if is_valid:
                logging.debug(f"SuperGlue verified match: {self.image_filenames[idx]} (ID: {idx}) with {num_inliers} inliers")
                return idx
        
        # Return the best match found (even if below threshold)
        if best_inliers > 0:
            logging.debug(f"Best SuperGlue match: {self.image_filenames[best_index]} (ID: {best_index}) with {best_inliers} inliers")
        else:
            logging.debug(f"No SuperGlue matches found, using CosPlace top match: {self.image_filenames[best_index]} (ID: {best_index})")
        
        return best_index

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID.
        This version uses CosPlace descriptors + SuperGlue verification.
        """
        # Ensure CosPlace model is available
        self._ensure_cosplace_model()
        if self.cosplace_model is None:
            logging.error("CosPlace model is not available; cannot build descriptor database.")
            return
        
        # Initialize SuperGlue models if available (for geometric verification)
        if self.superglue_available:
            self._ensure_superglue_models()
            if self.superpoint is not None:
                logging.info("SuperGlue verification enabled: will use CosPlace + SuperGlue for matching.")
            else:
                logging.warning("SuperGlue models failed to load. Falling back to CosPlace-only matching.")
        else:
            logging.info("SuperGlue not available. Using CosPlace-only matching.")

        # Build CosPlace descriptor database for all exploration images
        logging.info("Building CosPlace descriptor database...")
        if self.database is None:
            logging.info("Computing CosPlace descriptors for exploration images...")
            self.database = []
            self.image_filenames = []
            
            # Get all image files (both .jpg and .png), sorted naturally to preserve capture order
            # natsorted ensures proper numeric ordering (e.g., "1.jpg" < "10.jpg" < "2.jpg" becomes "1.jpg" < "2.jpg" < "10.jpg")
            all_files = os.listdir(self.save_dir)
            exploration_observation = natsorted(
                [x for x in all_files if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
            )
            
            if len(exploration_observation) == 0:
                logging.error(f"No image files (.jpg, .jpeg, .png) found in {self.save_dir}")
                return
            
            logging.info(f"Processing {len(exploration_observation)} exploration images (in folder order)")
            for img_name in tqdm(exploration_observation, desc="Processing images"):
                img_path = os.path.join(self.save_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Failed to read exploration image: {img_path}")
                    continue
                desc = self.compute_cosplace_descriptor_from_img(img)
                if desc is None:
                    logging.warning(f"CosPlace descriptor computation failed for: {img_path}")
                    continue
                # desc is (1,D); store as (D,)
                self.database.append(desc.squeeze(0))
                # Store the filename to maintain the mapping between index and actual file
                self.image_filenames.append(img_name)

            if len(self.database) == 0:
                logging.error("No CosPlace descriptors were computed; database is empty.")
                return

            self.database = np.stack(self.database, axis=0)

            # Build a BallTree for fast nearest neighbor search in descriptor space
            logging.info("Building BallTree on CosPlace descriptors...")
            self.tree = BallTree(self.database, leaf_size=64)


    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def get_filename_from_index(self, index):
        """
        Get the filename (e.g., "0.jpg" or "1.png") for a given database index.
        Returns the filename if available, otherwise returns a formatted string.
        """
        if self.image_filenames is not None and index < len(self.image_filenames):
            return self.image_filenames[index]
        else:
            # Fallback: try to construct from index
            return f"{index}.jpg"  # default assumption

    def display_goal_image(self):
        """
        Display the goal image that was found during navigation setup.
        """
        if self.goal is None:
            logging.warning("Goal has not been set yet.")
            return
        
        goal_filename = self.get_filename_from_index(self.goal)
        logging.info(f"Displaying goal image (ID: {self.goal}, File: {goal_filename})")
        self.display_img_from_id(self.goal, 'Goal Image')
        print(f'Goal Image: {goal_filename} (ID: {self.goal})')

    def display_next_best_view(self):
        """
        Display both the current view (where we are mapped) and the next best view.
        For debugging: shows current location in the map and the suggested next view.
        """
        # Get the neighbor of current FPV - this is where we currently are in the map
        current_index = self.get_neighbor(self.fpv)
        current_filename = self.get_filename_from_index(current_index)
        
        # Display the current view (where we are mapped in the folder)
        self.display_img_from_id(current_index, 'Current View (Map Location)')
        
        # Calculate next best view (a few frames ahead to avoid showing the exact same image)
        # Ensure we don't go out of bounds
        max_index = len(self.database) - 1 if self.database is not None else (len(self.image_filenames) - 1 if self.image_filenames is not None else current_index)
        next_index = min(current_index + 3, max_index)
        next_filename = self.get_filename_from_index(next_index)
        
        # Display the next best view
        self.display_img_from_id(next_index, 'Next Best View')
        
        # Print information with filenames
        goal_filename = self.get_filename_from_index(self.goal) if self.goal is not None else "N/A"
        print(f'Current View: {current_filename} (ID: {current_index}) || '
              f'Next Best View: {next_filename} (ID: {next_index}) || '
              f'Goal: {goal_filename} (ID: {self.goal})')

    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')
                    # Display the goal image immediately after it's found
                    self.display_goal_image()
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                # If 'g' key is pressed, display the goal image again
                if keys[pygame.K_g]:
                    self.display_goal_image()

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
