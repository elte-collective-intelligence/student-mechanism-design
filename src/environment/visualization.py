"""
Visualization module for Scotland Yard environment.

This module handles all rendering, plotting, and GIF generation for the game state,
including both regular game visualization and heatmap visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Optional, List


class GameVisualizer:
    """Handles visualization and rendering for the Scotland Yard game."""
    
    def __init__(self, vis_config: dict, logger):
        """
        Initialize the visualizer.
        
        Args:
            vis_config: Visualization configuration dictionary
            logger: Logger instance for debugging
        """
        self.vis_config = vis_config
        self.logger = logger
        self.visualize = vis_config["visualize_game"]
        self.visualize_heatmap = vis_config["visualize_heatmap"]
        
        # Matplotlib components
        self.fig = None
        self.ax = None
        self.pos = None
        self.node_colors = None
        self.node_collection = None
        self.edge_collection = None
        self.label_collection = None
        self.heatmap_colors = None
        self.hm_node_collection = None
        self.hm_edge_collection = None
        self.hm_label_collection = None
        
        # Image storage for GIFs
        self.run_images: List = []
        self.heatmap_images: List = []
        
        # Game state
        self.G: Optional[nx.Graph] = None
        self.board = None
        self.node_visits = None
        self.mrx_pos = None
        self.police_positions = None
        self.timestep = 0
        self.epoch = 0
        self.episode = 0
    
    def set_game_state(self, board, mrx_pos, police_positions, node_visits, timestep, epoch, episode):
        """
        Update the current game state for visualization.
        
        Args:
            board: Graph object with nodes and edges
            mrx_pos: MrX's position
            police_positions: List of police positions
            node_visits: Array of node visit counts
            timestep: Current timestep
            epoch: Current epoch
            episode: Current episode
        """
        self.board = board
        self.mrx_pos = mrx_pos
        self.police_positions = police_positions
        self.node_visits = node_visits
        self.timestep = timestep
        self.epoch = epoch
        self.episode = episode
    
    def initialize_render(self, reset=False):
        """
        Initialize the matplotlib plot for rendering the graph.
        
        Args:
            reset: Whether this is a reset initialization
        """
        self.logger.log("Initializing render plot.", level="debug")
        self.run_images = []
        self.heatmap_images = []
        
        # Create a NetworkX graph
        graph = self.board
        self.G = nx.Graph()
        num_nodes = graph.nodes.shape[0]
        self.G.add_nodes_from(range(num_nodes))
        edges = [tuple(edge) for edge in graph.edge_links]
        self.G.add_edges_from(edges)
        
        # Add edge weights if available
        if hasattr(graph, "edges") and graph.edges is not None:
            for idx, edge in enumerate(graph.edge_links):
                self.G.edges[tuple(edge)]["weight"] = graph.edges[idx]
        
        # Choose a layout
        self.pos = nx.kamada_kawai_layout(self.G)
        
        # Initialize matplotlib figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(
            f"Connected Graph Visualization at Timestep {self.timestep}", fontsize=16
        )
        
        # Draw nodes
        self.node_colors = ["lightblue"] * self.G.number_of_nodes()  # Default color
        self.heatmap_colors = ["lightblue"] * self.G.number_of_nodes()
        
        # Highlight MrX and police positions
        self.update_node_colors()
        
        self.node_collection = nx.draw_networkx_nodes(
            self.G,
            self.pos,
            ax=self.ax,
            node_size=700,
            node_color=self.node_colors,
            edgecolors="black",
        )
        
        # Draw edges
        self.edge_collection = nx.draw_networkx_edges(
            self.G, self.pos, ax=self.ax, width=2, edge_color="darkgray"
        )
        
        self.label_collection = nx.draw_networkx_labels(
            self.G,
            self.pos,
            ax=self.ax,
            font_size=10,
            font_family="sans-serif",
            font_color="white",
        )
        
        # Add legend
        red_patch = mpatches.Patch(color="red", label="MrX")
        blue_patch = mpatches.Patch(color="blue", label="Police")
        self.ax.legend(handles=[red_patch, blue_patch], loc="upper right")
        
        self.ax.axis("off")  # Hide the axes
        
        self.logger.log("Render plot initialized.", level="debug")
    
    def update_node_colors(self):
        """
        Update the colors of the nodes based on the positions of MrX and police agents.
        """
        # Reset all colors to default
        self.node_colors = ["gray"] * self.board.nodes.shape[0]
        self.heatmap_colors = []
        
        # Color MrX's position red
        if self.mrx_pos is not None:
            self.node_colors[self.mrx_pos] = "red"
        
        # Color police positions blue
        if self.police_positions is not None:
            for pos in self.police_positions:
                self.node_colors[pos] = "blue"
        
        # Generate heatmap colors
        if self.node_visits is not None:
            visit_max = np.amax([np.amax(self.node_visits), 1.0])
            for pos in range(self.node_visits.shape[0]):
                self.heatmap_colors.append(
                    ((self.node_visits[pos] / visit_max) * 0.9, 0, 0)
                )
    
    def render(self):
        """Renders the environment."""
        if self.fig is None or self.ax is None:
            # If render has not been initialized, initialize it
            self.initialize_render()
            return
        
        # Always update node colors if we're saving or visualizing
        if self.visualize or self.visualize_heatmap or self.vis_config.get("save_visualization", False):
            self.update_node_colors()
        
        # Capture game visualization frames
        if self.visualize or (self.vis_config.get("save_visualization", False) and not self.visualize_heatmap):
            # Update the node colors in the plot
            self.node_collection.set_color(self.node_colors)
            
            # Update the title
            self.ax.set_title(
                f"Epoch: {self.epoch}, Episode: {self.episode}, Timestep: {self.timestep}",
                fontsize=16,
            )
            
            # Redraw the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            buffer = self.fig.canvas.tostring_argb()
            w, h = self.fig.canvas.get_width_height()
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 4)[:, :, 1:]
            self.run_images.append(image)
            if self.visualize:  # Only log to plt if actively visualizing
                self.logger.log_plt("chart", plt)
        
        # Capture heatmap frames  
        if self.visualize_heatmap or (self.vis_config.get("save_visualization", False) and self.visualize_heatmap):
            self.node_collection.set_color(self.heatmap_colors)
            
            # Update the title
            self.ax.set_title(
                f"Epoch: {self.epoch}, Episode: {self.episode}, Timestep: {self.timestep}",
                fontsize=16,
            )
            
            # Redraw the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            buffer = self.fig.canvas.tostring_argb()
            w, h = self.fig.canvas.get_width_height()
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 4)[:, :, 1:]
            self.heatmap_images.append(image)
            self.logger.log_plt("heatmap", plt)
    
    def close_render(self):
        """Closes the matplotlib plot."""
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception as e:
                # Silently handle any matplotlib cleanup errors
                pass
            finally:
                self.fig = None
                self.ax = None
                self.node_collection = None
                self.edge_collection = None
                self.hm_node_collection = None
                self.hm_edge_collection = None
                self.logger.log("Render plot closed.", level="debug")
    
    def save_visualizations(self):
        """Save visualization GIFs if enabled."""
        if self.vis_config["save_visualization"] == True:
            
            if len(self.run_images) > 0:
                self.logger.log(
                    f"Saving GIF as run_epoch_{self.epoch}-episode_{self.episode+1}.gif",
                    level="info"
                )
                f, a = plt.subplots()
                img = a.imshow(self.run_images[0], animated=True)
                
                def update_gif(i):
                    img.set_array(self.run_images[i])
                    return (img,)
                
                animation_fig = animation.FuncAnimation(
                    f,
                    update_gif,
                    frames=len(self.run_images),
                    interval=400,
                    blit=True,
                    repeat_delay=10,
                )
                animation_fig.save(
                    self.vis_config["save_dir"]
                    + "/"
                    + f"run_epoch_{self.epoch}-episode_{self.episode+1}.gif"
                )
                plt.close(f)
                # Clear images after saving
                self.run_images = []
            
            if len(self.heatmap_images) > 0:
                self.logger.log(
                    f"Saving GIF as heatmap_epoch_{self.epoch}-episode_{self.episode+1}.gif",
                    level="info"
                )
                f, a = plt.subplots()
                img = a.imshow(self.heatmap_images[0], animated=True)
                
                def update_gif(i):
                    img.set_array(self.heatmap_images[i])
                    return (img,)
                
                animation_fig = animation.FuncAnimation(
                    f,
                    update_gif,
                    frames=len(self.heatmap_images),
                    interval=400,
                    blit=True,
                    repeat_delay=10,
                )
                animation_fig.save(
                    self.vis_config["save_dir"]
                    + "/"
                    + f"heatmap_epoch_{self.epoch}-episode_{self.episode+1}.gif"
                )
                plt.close(f)
                # Clear images after saving
                self.heatmap_images = []
