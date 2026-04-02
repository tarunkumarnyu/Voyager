"""
Autonomous Visual Navigation System
====================================
Combines:
1. CosPlace + SuperGlue localization (from baseline)
2. Visual Odometry + Graph building (from notebook)
3. A* pathfinding (NEW)
4. Autonomous execution with closed-loop control (NEW)

Author: Enhanced from nishant_baseline.py and cosplace_superglue__3_.ipynb
"""

# Import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import sys
import pickle
import json
from collections import deque
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
import heapq

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# NetworkX for graph building
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    logging.info("NetworkX available - full graph features enabled")
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available - graph features disabled")

# Try to import SuperPoint and SuperGlue
try:
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


class AutonomousNavigator(Player):
    """
    Autonomous navigation player that combines localization, mapping, and planning.
    """
    
    def __init__(self):
        # Initialize base class
        super(AutonomousNavigator, self).__init__()
        
        # Display variables
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        
        # Data directories
        self.save_dir = "data/images_subsample/"
        self.data_info_path = "data/data_info.json"  # Action labels
        
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # CosPlace model and transform
        self.cosplace_model = None
        self.cosplace_transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # Localization data
        self.database = None  # CosPlace descriptors (N, 512)
        self.tree = None  # BallTree for KNN search
        self.image_filenames = None  # List of image filenames
        
        # Visual Odometry and Graph data
        self.centers = None  # VO trajectory (N, 3) - [x, y, z] positions
        self.G = None  # Visual place graph (NetworkX)
        self.G_nav = None  # Navigation graph with action labels (NetworkX DiGraph)
        self.loop_closures = []  # List of detected loop closures
        
        # SuperGlue models
        self.superpoint = None
        self.superglue = None
        self.superglue_available = SuperPoint is not None and SuperGlue is not None
        
        # Camera intrinsics
        self.K = np.array([[92.,   0., 160.],
                          [ 0.,  92., 120.],
                          [ 0.,   0.,  1. ]], dtype=np.float64)
        
        # Localization parameters
        self.cosplace_top_k = 5
        self.min_inliers = 30
        
        # Navigation state
        self.goal = None  # Goal node ID
        self.current_node = None  # Current node ID in graph
        self.autonomous_mode = False  # Toggle autonomous navigation
        self.planned_path = None  # List of node IDs from A*
        self.action_queue = deque()  # Queue of actions to execute
        self.frame_count = 0
        
        # Re-planning parameters
        self.replan_threshold = 3  # Re-plan if more than 3 nodes off track
        self.stuck_threshold = 5  # Consider stuck if same node for 5 frames
        self.stuck_counter = 0
        self.last_node = None
        
        # Visited locations for loop closure
        self.visited_locations = []
        self.loop_closure_threshold = 50
        
        # Statistics
        self.total_replans = 0
        self.total_actions = 0
        
    # ============================================================================
    # SUPERGLUE & LOCALIZATION (from baseline)
    # ============================================================================
    
    def load_superglue_models(self):
        """Initialize SuperPoint and SuperGlue models."""
        if not self.superglue_available:
            logging.warning("SuperGlue models not available.")
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
        """Lazy-initialize SuperGlue models."""
        if self.superglue_available and self.superpoint is None:
            self.superpoint, self.superglue = self.load_superglue_models()
    
    def match_superglue(self, img1_gray, img2_gray):
        """Run SuperPoint + SuperGlue between two grayscale images."""
        if not self.superglue_available or self.superpoint is None or self.superglue is None:
            return None, None
        
        t1 = torch.from_numpy(img1_gray).float() / 255.0
        t2 = torch.from_numpy(img2_gray).float() / 255.0
        
        t1 = t1[None, None].to(self.device)
        t2 = t2[None, None].to(self.device)
        
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
        """Estimate relative motion using Essential matrix + recoverPose."""
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
        """Verify a match between two images using SuperGlue."""
        if not self.superglue_available:
            return 0, False
        
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
    
    # ============================================================================
    # COSPLACE MODEL (from baseline)
    # ============================================================================
    
    def load_cosplace_model(self):
        """Load pretrained CosPlace model."""
        try:
            logging.info("Loading CosPlace model from torch.hub...")
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
        """Lazy-initialize CosPlace model."""
        if self.cosplace_model is None:
            self.cosplace_model = self.load_cosplace_model()
    
    def compute_cosplace_descriptor_from_img(self, img_bgr):
        """Compute L2-normalized CosPlace descriptor for a BGR image."""
        self._ensure_cosplace_model()
        if self.cosplace_model is None:
            logging.error("CosPlace model not available.")
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGB")
        
        try:
            t = self.cosplace_transform(pil_img)
        except Exception as e:
            logging.error(f"Failed to transform image: {e}")
            return None
        
        t = t.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feats = self.cosplace_model(t)
            feats = F.normalize(feats, p=2, dim=1)
        
        return feats.cpu().numpy().astype(np.float32)
    
    # ============================================================================
    # VISUAL ODOMETRY (from notebook)
    # ============================================================================
    
    def run_vo_superglue(self, img_paths, scale=1.0):
        """
        Run visual odometry using SuperGlue to build trajectory.
        
        Returns:
            centers: (N, 3) array of camera positions
            used_paths: List of image paths successfully processed
            skipped_images: List of skipped image indices
        """
        self._ensure_superglue_models()
        
        if not self.superglue_available:
            logging.warning("SuperGlue not available - cannot run VO. Using sequential ordering.")
            # Return dummy trajectory based on sequential order
            N = len(img_paths)
            centers = np.zeros((N, 3))
            centers[:, 2] = np.arange(N)  # Just place along Z axis
            return centers, img_paths, []
        
        logging.info("Running Visual Odometry with SuperGlue...")
        
        # Current pose in world frame
        R_world = np.eye(3)
        t_world = np.zeros((3, 1))
        
        centers = [t_world.flatten()]
        used_paths = [img_paths[0]]
        skipped_images = []
        
        prev_gray = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
        if prev_gray is None:
            logging.error(f"Could not load first image: {img_paths[0]}")
            return None, None, None
        
        for i in tqdm(range(1, len(img_paths)), desc="VO Progress"):
            curr_gray = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
            if curr_gray is None:
                logging.warning(f"Skipping {img_paths[i]}")
                skipped_images.append(i)
                continue
            
            # Match features
            pts1, pts2 = self.match_superglue(prev_gray, curr_gray)
            
            if pts1 is None or len(pts1) < 8:
                logging.warning(f"Not enough matches at frame {i}")
                skipped_images.append(i)
                continue
            
            # Estimate motion
            R, t, mask = self.estimate_motion(pts1, pts2)
            
            if R is None:
                logging.warning(f"Motion estimation failed at frame {i}")
                skipped_images.append(i)
                continue
            
            # Update world pose
            t_scaled = t * scale
            t_world = t_world + R_world @ t_scaled
            R_world = R_world @ R
            
            centers.append(t_world.flatten())
            used_paths.append(img_paths[i])
            prev_gray = curr_gray
        
        centers = np.array(centers)
        logging.info(f"VO completed: {len(centers)} poses, {len(skipped_images)} skipped")
        
        return centers, used_paths, skipped_images
    
    # ============================================================================
    # GRAPH BUILDING (from notebook)
    # ============================================================================
    
    def build_place_graph(self, descs, centers, image_names, k=10):
        """
        Build NetworkX graph with visual similarity and sequential edges.
        
        Returns:
            G: NetworkX Graph with nodes and edges
        """
        if not NETWORKX_AVAILABLE:
            logging.error("NetworkX not available - cannot build graph")
            return None
        
        M = len(descs)
        logging.info(f"Building place graph for {M} nodes...")
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(M):
            G.add_node(i,
                      name=image_names[i],
                      path=os.path.join(self.save_dir, image_names[i]),
                      x=centers[i, 0],
                      y=centers[i, 1],
                      z=centers[i, 2])
        
        # Add sequential edges
        for i in range(M - 1):
            G.add_edge(i, i + 1,
                      sequence=True,
                      cosplace=False,
                      weight=1.0)
        
        # Add CosPlace KNN edges
        logging.info(f"Adding visual similarity edges (k={k})...")
        for i in tqdm(range(M), desc="KNN edges"):
            dists = np.linalg.norm(descs - descs[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]  # Exclude self
            
            for j in neighbors:
                if not G.has_edge(i, j):
                    G.add_edge(i, j,
                              sequence=False,
                              cosplace=True,
                              weight=dists[j])
        
        logging.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def load_data_info(self):
        """Load action labels from data_info.json."""
        if not os.path.exists(self.data_info_path):
            logging.warning(f"data_info.json not found at {self.data_info_path}")
            return None
        
        try:
            with open(self.data_info_path, 'r') as f:
                data_info = json.load(f)
            data_info = sorted(data_info, key=lambda d: d.get("step", 0))
            logging.info(f"Loaded {len(data_info)} entries from data_info.json")
            return data_info
        except Exception as e:
            logging.error(f"Failed to load data_info.json: {e}")
            return None
    
    def build_action_edges(self, data_info):
        """
        Build action-labeled edges from data_info.json.
        
        Returns:
            action_edges: List of (u, v, actions) tuples
        """
        if data_info is None:
            return []
        
        # Build mappings
        step_to_image = {}
        step_to_action = {}
        image_to_node = {name: i for i, name in enumerate(self.image_filenames)}
        
        for entry in data_info:
            step = int(entry["step"])
            img_name = entry["image"]
            actions = entry.get("actions", [])
            
            step_to_image[step] = img_name
            step_to_action[step] = actions
        
        # Build action edges
        steps = sorted(step_to_image.keys())
        action_edges = []
        
        for i in range(len(steps) - 1):
            s = steps[i]
            s_next = steps[i + 1]
            
            img_s = step_to_image.get(s)
            img_next = step_to_image.get(s_next)
            
            if img_s is None or img_next is None:
                continue
            
            u = image_to_node.get(img_s)
            v = image_to_node.get(img_next)
            
            if u is None or v is None:
                continue
            
            actions = step_to_action.get(s, [])
            action_edges.append((u, v, actions))
        
        logging.info(f"Built {len(action_edges)} action edges")
        return action_edges
    
    def attach_actions_to_graph(self, G, action_edges):
        """
        Create directed navigation graph with action labels.
        
        Returns:
            G_nav: NetworkX DiGraph with action-labeled edges
        """
        if not NETWORKX_AVAILABLE or G is None:
            return None
        
        G_nav = nx.DiGraph()
        
        # Copy nodes
        for n, attrs in G.nodes(data=True):
            G_nav.add_node(n, **attrs)
        
        # Add action edges
        for u, v, actions in action_edges:
            if G_nav.has_edge(u, v):
                existing = G_nav[u][v].get("actions", [])
                merged = list(set(existing + actions))
                G_nav[u][v]["actions"] = merged
            else:
                G_nav.add_edge(u, v, actions=actions)
        
        # Also add reverse sequential edges (for backtracking)
        for i in range(len(self.image_filenames) - 1):
            if not G_nav.has_edge(i+1, i):
                G_nav.add_edge(i+1, i, actions=["BACKWARD"])
        
        logging.info(f"Navigation graph: {G_nav.number_of_nodes()} nodes, {G_nav.number_of_edges()} edges")
        return G_nav
    
    # ============================================================================
    # A* PATHFINDING (NEW!)
    # ============================================================================
    
    def astar_navigation(self, start, goal):
        """
        A* pathfinding on navigation graph.
        
        Args:
            start: Start node ID
            goal: Goal node ID
            
        Returns:
            path: List of node IDs from start to goal, or None if no path
        """
        if self.G_nav is None:
            logging.error("Navigation graph not available")
            return None
        
        if start not in self.G_nav or goal not in self.G_nav:
            logging.error(f"Start {start} or goal {goal} not in graph")
            return None
        
        def heuristic(node):
            """Euclidean distance in VO space."""
            if self.centers is None:
                return abs(goal - node)  # Fallback to sequential distance
            
            node_pos = self.centers[node]
            goal_pos = self.centers[goal]
            return np.linalg.norm(node_pos - goal_pos)
        
        def edge_cost(u, v):
            """Cost based on actions and edge type."""
            if not self.G_nav.has_edge(u, v):
                return float('inf')
            
            actions = self.G_nav[u][v].get('actions', [])
            base_cost = len(actions) if actions else 1
            
            # Penalize turns
            turn_penalty = 0
            if 'LEFT' in actions or 'RIGHT' in actions:
                turn_penalty = 0.5
            
            # Prefer sequential edges
            if self.G is not None and self.G.has_edge(u, v) and self.G[u][v].get('sequence', False):
                sequential_bonus = -0.2
            else:
                sequential_bonus = 0
            
            return base_cost + turn_penalty + sequential_bonus
        
        # A* algorithm
        open_set = [(0, start, [])]
        visited = set()
        g_score = {start: 0}
        
        while open_set:
            f, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path + [current]
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore neighbors
            for neighbor in self.G_nav.successors(current):
                tentative_g = g_score[current] + edge_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor, path + [current]))
        
        logging.warning(f"No path found from {start} to {goal}")
        return None
    
    def extract_action_sequence(self, path):
        """
        Extract action sequence from a path.
        
        Args:
            path: List of node IDs
            
        Returns:
            actions: List of action strings
        """
        if path is None or len(path) < 2:
            return []
        
        actions = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G_nav.has_edge(u, v):
                edge_actions = self.G_nav[u][v].get('actions', [])
                actions.extend(edge_actions)
            else:
                logging.warning(f"No edge from {u} to {v}")
                actions.append("FORWARD")  # Fallback
        
        return actions
    
    # ============================================================================
    # LOCALIZATION (from baseline)
    # ============================================================================
    
    def get_neighbor(self, fpv):
        """
        Find which exploration image best matches current view.
        
        Returns:
            index: Node ID in graph
        """
        query_desc = self.compute_cosplace_descriptor_from_img(fpv)
        if query_desc is None:
            return self.current_node if self.current_node is not None else 0
        
        if self.tree is None:
            logging.warning("BallTree not available; returning current node or 0")
            return self.current_node if self.current_node is not None else 0
        
        distances, indices = self.tree.query(query_desc, k=self.cosplace_top_k)
        candidates = indices[0]  # Shape: (k,)
        
        # If SuperGlue available, verify with geometry
        if self.superglue_available:
            self._ensure_superglue_models()
            if self.superpoint is not None and self.superglue is not None:
                best_match = None
                max_inliers = 0
                
                for candidate_idx in candidates:
                    candidate_idx = int(candidate_idx)  # Convert to int
                    if candidate_idx >= len(self.image_filenames):
                        continue
                    
                    candidate_path = os.path.join(self.save_dir, self.image_filenames[candidate_idx])
                    candidate_img = cv2.imread(candidate_path)
                    
                    if candidate_img is None:
                        continue
                    
                    num_inliers, is_valid = self.verify_match_with_superglue(fpv, candidate_img)
                    
                    if is_valid and num_inliers > max_inliers:
                        max_inliers = num_inliers
                        best_match = candidate_idx
                
                if best_match is not None:
                    return best_match
        
        # Fallback to closest CosPlace match
        return int(candidates[0])
    
    # ============================================================================
    # PRE-NAVIGATION SETUP
    # ============================================================================
    
    def pre_navigation(self):
        """Called by game framework before navigation phase starts."""
        super(AutonomousNavigator, self).pre_navigation()
        self.pre_nav_compute_cv()
    
    def pre_nav_compute_cv(self):
        """Build database, trajectory, and graphs before navigation."""
        logging.info("="*70)
        logging.info("PRE-NAVIGATION SETUP")
        logging.info("="*70)
        
        # 1. Load image filenames
        self.image_filenames = natsorted([
            f for f in os.listdir(self.save_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_filenames) == 0:
            logging.error(f"No images found in {self.save_dir}")
            return
        
        logging.info(f"Found {len(self.image_filenames)} images")
        
        # 2. Load or compute CosPlace descriptors
        cache_file = os.path.join(self.save_dir, "cosplace_descriptors.pkl")
        
        if os.path.exists(cache_file):
            logging.info("Loading CosPlace descriptors from cache...")
            with open(cache_file, 'rb') as f:
                self.database = pickle.load(f)
        else:
            logging.info("Computing CosPlace descriptors...")
            self._ensure_cosplace_model()
            
            descriptors = []
            for filename in tqdm(self.image_filenames, desc="Extracting descriptors"):
                img_path = os.path.join(self.save_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not load {filename}")
                    descriptors.append(np.zeros(512, dtype=np.float32))
                    continue
                
                desc = self.compute_cosplace_descriptor_from_img(img)
                if desc is None:
                    descriptors.append(np.zeros(512, dtype=np.float32))
                else:
                    descriptors.append(desc[0])
            
            self.database = np.array(descriptors)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.database, f)
            logging.info(f"Saved descriptors to {cache_file}")
        
        logging.info(f"Database shape: {self.database.shape}")
        
        # 3. Build KNN search tree
        self.tree = BallTree(self.database, metric='euclidean')
        logging.info("Built BallTree for KNN search")
        
        # 4. Run Visual Odometry
        vo_cache_file = os.path.join(self.save_dir, "vo_trajectory.pkl")
        
        if os.path.exists(vo_cache_file):
            logging.info("Loading VO trajectory from cache...")
            with open(vo_cache_file, 'rb') as f:
                self.centers = pickle.load(f)
        else:
            img_paths = [os.path.join(self.save_dir, f) for f in self.image_filenames]
            self.centers, used_paths, skipped = self.run_vo_superglue(img_paths, scale=1.0)
            
            if self.centers is not None:
                with open(vo_cache_file, 'wb') as f:
                    pickle.dump(self.centers, f)
                logging.info(f"Saved VO trajectory to {vo_cache_file}")
        
        if self.centers is not None:
            logging.info(f"VO trajectory shape: {self.centers.shape}")
        
        # 5. Build place graph
        if NETWORKX_AVAILABLE and self.centers is not None:
            graph_cache_file = os.path.join(self.save_dir, "place_graph.pkl")
            
            if os.path.exists(graph_cache_file):
                logging.info("Loading place graph from cache...")
                with open(graph_cache_file, 'rb') as f:
                    self.G = pickle.load(f)
            else:
                self.G = self.build_place_graph(
                    self.database,
                    self.centers,
                    self.image_filenames,
                    k=10
                )
                
                if self.G is not None:
                    with open(graph_cache_file, 'wb') as f:
                        pickle.dump(self.G, f)
                    logging.info(f"Saved place graph to {graph_cache_file}")
        
        # 6. Load action labels and build navigation graph
        if self.G is not None:
            data_info = self.load_data_info()
            if data_info is not None:
                action_edges = self.build_action_edges(data_info)
                self.G_nav = self.attach_actions_to_graph(self.G, action_edges)
            else:
                logging.warning("No data_info.json - navigation graph will have no action labels")
                # Create basic navigation graph without action labels
                self.G_nav = self.G.to_directed()
        
        # 7. Initialize SuperGlue models
        self._ensure_superglue_models()
        
        logging.info("="*70)
        logging.info("PRE-NAVIGATION SETUP COMPLETE")
        logging.info("="*70)
        
        if self.G_nav is not None:
            logging.info(f"✓ Navigation graph ready: {self.G_nav.number_of_nodes()} nodes, "
                        f"{self.G_nav.number_of_edges()} edges")
            logging.info("✓ Autonomous navigation ENABLED")
        else:
            logging.warning("✗ Navigation graph not available - autonomous mode limited")
    
    # ============================================================================
    # AUTONOMOUS NAVIGATION CONTROL
    # ============================================================================
    
    def plan_to_goal(self):
        """Plan path from current position to goal."""
        if self.current_node is None or self.goal is None:
            logging.warning("Cannot plan: current_node or goal not set")
            return False
        
        if self.G_nav is None:
            logging.error("Navigation graph not available")
            return False
        
        logging.info(f"Planning path from {self.current_node} to {self.goal}...")
        
        self.planned_path = self.astar_navigation(self.current_node, self.goal)
        
        if self.planned_path is None:
            logging.error(f"No path found from {self.current_node} to {self.goal}")
            return False
        
        # Extract actions
        actions = self.extract_action_sequence(self.planned_path)
        self.action_queue = deque(actions)
        
        self.total_replans += 1
        
        logging.info(f"Path planned: {len(self.planned_path)} nodes, {len(actions)} actions")
        logging.info(f"Path: {self.planned_path[:10]}{'...' if len(self.planned_path) > 10 else ''}")
        
        return True
    
    def should_replan(self):
        """Check if we need to re-plan."""
        if self.planned_path is None or len(self.planned_path) == 0:
            return True
        
        # Check if we're still on the planned path
        try:
            current_pos_in_path = self.planned_path.index(self.current_node)
            # We're on path - check if we're too far behind
            if current_pos_in_path > self.replan_threshold:
                logging.info(f"Behind on path (position {current_pos_in_path}), replanning...")
                return True
        except ValueError:
            # Not on path at all
            logging.info(f"Off path (current: {self.current_node}, path: {self.planned_path[:5]}...), replanning...")
            return True
        
        return False
    
    def check_if_stuck(self):
        """Check if robot is stuck at same location."""
        if self.current_node == self.last_node:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_node = self.current_node
        
        if self.stuck_counter >= self.stuck_threshold:
            logging.warning(f"Stuck at node {self.current_node} for {self.stuck_counter} frames")
            return True
        
        return False
    
    def autonomous_act(self):
        """
        Autonomous action selection with closed-loop control.
        
        Returns:
            Action: Next action to execute
        """
        # Localize current position
        self.current_node = self.get_neighbor(self.fpv)
        
        # Check if reached goal
        if self.current_node == self.goal:
            logging.info("="*70)
            logging.info(f"GOAL REACHED! Node {self.goal}")
            logging.info(f"Total actions: {self.total_actions}")
            logging.info(f"Total replans: {self.total_replans}")
            logging.info("="*70)
            self.autonomous_mode = False
            return Action.CHECKIN
        
        # Check if stuck
        if self.check_if_stuck():
            logging.info("Attempting recovery: turning right")
            self.stuck_counter = 0
            return Action.RIGHT
        
        # Check if need to replan
        if len(self.action_queue) == 0 or self.should_replan():
            success = self.plan_to_goal()
            if not success:
                logging.error("Planning failed - stopping autonomous mode")
                self.autonomous_mode = False
                return Action.IDLE
        
        # Execute next action
        if len(self.action_queue) > 0:
            action_str = self.action_queue.popleft()
            self.total_actions += 1
            
            # Convert string to Action
            action_map = {
                "FORWARD": Action.FORWARD,
                "LEFT": Action.LEFT,
                "RIGHT": Action.RIGHT,
                "BACKWARD": Action.BACKWARD,
            }
            
            action = action_map.get(action_str, Action.FORWARD)
            
            if self.total_actions % 10 == 0:
                logging.info(f"Action {self.total_actions}: {action_str} | "
                           f"Current: {self.current_node} | Goal: {self.goal} | "
                           f"Queue: {len(self.action_queue)} actions left")
            
            return action
        
        return Action.IDLE
    
    # ============================================================================
    # GAME INTERFACE
    # ============================================================================
    
    def reset(self):
        """Reset player state."""
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        
        self.current_node = None
        self.goal = None
        self.autonomous_mode = False
        self.planned_path = None
        self.action_queue = deque()
        self.frame_count = 0
        self.total_replans = 0
        self.total_actions = 0
        self.stuck_counter = 0
        self.last_node = None
        
        self.visited_locations = []
        self.loop_closures = []
        
        pygame.init()
        
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }
    
    def act(self):
        """Handle player actions."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            
            if event.type == pygame.KEYDOWN:
                # Toggle autonomous mode with 'a' key
                if event.key == pygame.K_a:
                    self.autonomous_mode = not self.autonomous_mode
                    if self.autonomous_mode:
                        logging.info("="*70)
                        logging.info("AUTONOMOUS MODE ENABLED")
                        logging.info("="*70)
                        # Initial localization
                        self.current_node = self.get_neighbor(self.fpv)
                        logging.info(f"Starting position: Node {self.current_node}")
                    else:
                        logging.info("AUTONOMOUS MODE DISABLED - Switching to manual control")
                    return Action.IDLE
                
                # Display info with 'i' key
                if event.key == pygame.K_i:
                    self.display_status()
                    return Action.IDLE
                
                # Manual control if not autonomous
                if not self.autonomous_mode:
                    if event.key in self.keymap:
                        self.last_act = self.keymap[event.key]
                        return self.last_act
        
        # Autonomous mode
        if self.autonomous_mode:
            return self.autonomous_act()
        
        return Action.IDLE
    
    def see(self, fpv):
        """Process first-person view."""
        if fpv is None or len(fpv.shape) < 3:
            return
        
        self.fpv = fpv
        self.frame_count += 1
        
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        
        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image
        
        pygame.display.set_caption("Autonomous Navigator")
        
        if self._state:
            if self._state[1] == Phase.NAVIGATION:
                # Set goal on first frame
                if self.goal is None:
                    targets = self.get_target_images()
                    if len(targets) > 0:
                        index = self.get_neighbor(targets[0])
                        self.goal = index
                        logging.info(f"Goal set to node {self.goal}")
        
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
    
    def display_status(self):
        """Display current navigation status."""
        print("\n" + "="*70)
        print("NAVIGATION STATUS")
        print("="*70)
        print(f"Autonomous Mode: {'ENABLED' if self.autonomous_mode else 'DISABLED'}")
        print(f"Current Node: {self.current_node}")
        print(f"Goal Node: {self.goal}")
        print(f"Frame: {self.frame_count}")
        print(f"Total Actions: {self.total_actions}")
        print(f"Total Replans: {self.total_replans}")
        
        if self.planned_path:
            print(f"Planned Path Length: {len(self.planned_path)} nodes")
            print(f"Actions in Queue: {len(self.action_queue)}")
            print(f"Next 5 nodes: {self.planned_path[:5]}")
        else:
            print("No active plan")
        
        if self.G_nav:
            print(f"\nGraph: {self.G_nav.number_of_nodes()} nodes, {self.G_nav.number_of_edges()} edges")
        
        print("="*70)
        print("Controls:")
        print("  'a' - Toggle autonomous mode")
        print("  'i' - Display this status")
        print("  Arrow keys - Manual control (when autonomous disabled)")
        print("  Space - Check in")
        print("  Escape - Quit")
        print("="*70 + "\n")


if __name__ == "__main__":
    import vis_nav_game
    
    player = AutonomousNavigator()
    
    try:
        vis_nav_game.play(the_player=player)
    finally:
        # Save statistics
        if player.autonomous_mode or player.total_actions > 0:
            stats = {
                'total_actions': player.total_actions,
                'total_replans': player.total_replans,
                'goal_reached': player.current_node == player.goal if player.goal else False,
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_file = f"navigation_stats_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\nNavigation statistics saved to: {stats_file}")
