import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

def create_architecture_diagram():
    """Create a clean, professional architecture diagram with perfect spacing."""
    
    # Create figure with extra width for spacious layout
    fig, ax = plt.subplots(figsize=(22, 16), dpi=200)
    
    # Set axis limits
    ax.set_xlim(0, 24)
    ax.set_ylim(-3, 17)
    ax.axis('off')
    ax.set_aspect('equal')
    
    COLORS = {
        'user': '#B3E5FC',        # Light blue
        'router': '#C8E6C9',      # Light green
        'decision': '#FFF9C4',    # Light yellow
        'analysis1': '#FFCDD2',   # Light red/pink
        'analysis2': '#F8BBD0',   # Light pink
        'eda': '#E1BEE7',         # Light purple
        'forecast': '#BBDEFB',    # Sky blue
        'stage': '#B3E5FC',       # Light cyan
        'report': '#E0E0E0',      # Light grey
        'badge': '#C8E6C9',       # Light green for ReAct
        'arrow': '#424242'        # Dark grey for arrows
    }
    
    def draw_rounded_box(x, y, w, h, text, color, fontsize=10, bold=True, edge_color='#424242', lw=2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.3",
                           facecolor=color, edgecolor=edge_color, linewidth=lw)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)
        return x + w/2, y + h/2, x, y, w, h # Return center and dims
    
    def draw_diamond(cx, cy, size, text, color):
        half = size / 2
        diamond = plt.Polygon([(cx, cy + half), (cx + half, cy), (cx, cy - half), (cx - half, cy)],
                            facecolor=color, edgecolor='#424242', linewidth=2)
        ax.add_patch(diamond)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=9, fontweight='bold')
        return cx, cy, half
    
    def draw_react_badge(x, y):
        badge = FancyBboxPatch((x, y), 0.9, 0.35, boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['badge'], edgecolor='#388E3C', linewidth=1.5)
        ax.add_patch(badge)
        ax.text(x + 0.45, y + 0.175, "ReAct", ha='center', va='center', fontsize=7, fontweight='bold', color='#1B5E20')
    
    def draw_arrow_path(points, text=None, text_pos=None):
        """Draw a multi-segment line with an arrow at the end."""
        # Draw all segments except arrow head
        for i in range(len(points)-1):
            if i == len(points)-2: continue # path handled by arrow
            p1 = points[i]
            p2 = points[i+1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#616161', lw=1.5)
            
        # Draw final arrow segment
        last_start = points[-2]
        last_end = points[-1]
        ax.annotate('', xy=last_end, xytext=last_start,
                   arrowprops=dict(arrowstyle='->', color='#616161', lw=1.5, mutation_scale=15))
        
        if text and text_pos:
            ax.text(text_pos[0], text_pos[1], text, fontsize=9, color='#616161', 
                   ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))

    # ============ CENTER CONSTANTS ============
    CX_USER = 12
    CX_LEFT = 4.5
    CX_MID = 12
    CX_RIGHT = 19.5
    
    # ============ NODES ============
    
    # Top Layer
    draw_rounded_box(CX_USER - 2, 14.5, 4, 1.2, "User\n(Natural Language Query)", COLORS['user'])
    draw_rounded_box(CX_USER - 3, 12, 6, 1.5, "Conversation Agent\n(Intent Recognition + Routing)", COLORS['router'])
    draw_react_badge(CX_USER + 2.2, 13.1)
    
    draw_diamond(CX_MID, 9.5, 2.0, "What\ndoes user\nwant?", COLORS['decision'])
    
    # Pathway Labels - Moved higher and distinct
    ax.text(CX_LEFT, 8.5, "Analysis Capabilities", ha='center', fontweight='bold', color='#C62828', style='italic', fontsize=11)
    ax.text(CX_MID, 8.5, "Exploration", ha='center', fontweight='bold', color='#6A1B9A', style='italic', fontsize=11)
    ax.text(CX_RIGHT, 8.5, "Forecasting Pipeline", ha='center', fontweight='bold', color='#1565C0', style='italic', fontsize=11)
    
    # --- LEFT PATH (Analysis) ---
    # Moved down to ensure arrows don't cross text
    LEVEL_1_Y = 5.0
    
    draw_rounded_box(1.5, LEVEL_1_Y, 3.5, 1.5, "Get Summaries\n(Stage 1)", COLORS['analysis1'])
    draw_rounded_box(6.0, LEVEL_1_Y, 3.5, 1.5, "Propose Tasks\n(Stage 2)", COLORS['analysis2'])
    
    # --- MIDDLE PATH (EDA) ---
    draw_rounded_box(CX_MID - 2, LEVEL_1_Y, 4, 1.8, "EDA Agent\n(Code Gen)", COLORS['eda'])
    draw_react_badge(CX_MID + 1.2, 5.8+0.5)
    draw_rounded_box(CX_MID - 2, 2.0, 4, 1.2, "Statistics, Plots,\nCorrelations", '#EDE7F6')
    
    # --- RIGHT PATH (Forecasting) ---
    draw_rounded_box(CX_RIGHT - 2, LEVEL_1_Y, 4, 1.5, "Forecasting\nManager", COLORS['forecast'])
    
    # Grid - Spaced out properly
    # Using specific coords for clarity
    
    b3 = draw_rounded_box(15.5, 2.5, 2.2, 0.9, "3: Plan", COLORS['stage'], fontsize=9)
    b3b = draw_rounded_box(18.2, 2.5, 2.2, 0.9, "3B: Prep", COLORS['stage'], fontsize=9)
    b35a = draw_rounded_box(20.9, 2.5, 2.2, 0.9, "3.5A: Method", COLORS['stage'], fontsize=9)
    
    b35b = draw_rounded_box(15.5, 0.8, 2.2, 0.9, "3.5B: Bench", COLORS['stage'], fontsize=9)
    b4 = draw_rounded_box(18.2, 0.8, 2.2, 0.9, "4: Execute", COLORS['stage'], fontsize=9)
    b5 = draw_rounded_box(20.9, 0.8, 2.2, 0.9, "5: Visualize", COLORS['stage'], fontsize=9)
    
    # Stage 6 below
    draw_rounded_box(16.5, -1.0, 6, 1.0, "Stage 6: Generate Final Report", COLORS['report'])
    draw_react_badge(21.8, -0.7)
    
    
    # ============ CONNECTIONS (Manhattan Style) ============
    
    # Agent <-> User
    ax.annotate('', xy=(12, 13.6), xytext=(12, 14.4), arrowprops=dict(arrowstyle='<->', color='#1E88E5', lw=2.5))
    
    # Agent -> Decision
    ax.annotate('', xy=(12, 10.6), xytext=(12, 11.9), arrowprops=dict(arrowstyle='->', color='#616161', lw=2))

    # Decision -> Left Branch
    # Go LEFT from Diamond Center, then DOWN
    # Diamond Center is (12, 9.5), Left Tip is (11, 9.5)
    # Common trunk left
    ax.plot([10.9, 3.25], [9.5, 9.5], color='#616161', lw=1.5)
    
    # Branch to Summaries (x=3.25)
    # Arrow path: (3.25, 9.5) -> (3.25, 6.6)
    draw_arrow_path([(3.25, 9.5), (3.25, 6.6)], '"show summaries"', (3.25, 7.5))
    
    # Branch to Proposals (x=7.75)
    # Arrow path: (10.9, 9.5) -> (7.75, 9.5) -> (7.75, 6.6)
    # Start separate line from trunk
    draw_arrow_path([(10.9, 9.5), (7.75, 9.5), (7.75, 6.6)], '"propose"', (7.75, 7.5))
    
    # Decision -> Center (EDA)
    # Arrow path: Down from diamond bottom
    # Bottom tip is (12, 8.5)
    # Avoid "Exploration" label at 8.0
    draw_arrow_path([(12, 8.4), (12, 6.9)], '"explore data"', (12.8, 7.6)) # Label beside arrow
    
    # EDA internal
    draw_arrow_path([(12, 4.9), (12, 3.3)])
    
    # Decision -> Right Branch (Forecasting)
    # Go RIGHT from Diamond Center
    # Diamond center (12, 9.5), Right tip (13, 9.5)
    draw_arrow_path([(13.1, 9.5), (19.5, 9.5), (19.5, 6.1)], '"run task"', (19.5, 7.5))
    
    # Manager -> 3: Plan
    # Down from Manager (19.5, 4.9) -> Left -> Down
    # Actually, Manager bottom is 5.0 -> 3 Top is 3.5ish
    # Simple path: (19.5, 4.9) -> (19.5, 4.2) -> (16.6, 4.2) -> (16.6, 3.5)
    draw_arrow_path([(19.5, 4.9), (19.5, 4.2), (16.6, 4.2), (16.6, 3.5)])
    
    # 3 -> 3B
    draw_arrow_path([(17.7, 2.95), (18.2, 2.95)]) 
    # 3B -> 3.5A
    draw_arrow_path([(20.4, 2.95), (20.9, 2.95)])
    
    # 3.5A -> 3.5B (The Wrap Around - Widened to avoid text)
    # Right of 3.5A is 23.1
    draw_arrow_path([(23.1, 2.95), (23.8, 2.95), (23.8, 0.4), (16.6, 0.4), (16.6, 0.7)])
    
    # 3.5B -> 4
    draw_arrow_path([(17.7, 1.25), (18.2, 1.25)])
    # 4 -> 5
    draw_arrow_path([(20.4, 1.25), (20.9, 1.25)])
    # 5 -> 6
    draw_arrow_path([(22.0, 0.8), (22.0, 0.4), (19.5, 0.4), (19.5, 0.1)])

    # ============ LEGEND (Moved Left) ============
    leg_box = FancyBboxPatch((0.5, 0.5), 5, 2.5, boxstyle="round,pad=0.1", fc='white', ec='#BDBDBD', lw=1)
    ax.add_patch(leg_box)
    ax.text(3, 2.5, "Legend", ha='center', fontweight='bold')
    
    ax.annotate('', xy=(1.5, 2.0), xytext=(4.5, 2.0), arrowprops=dict(arrowstyle='<->', color='#1E88E5', lw=2))
    ax.text(3, 1.7, "User / Agent Interaction", ha='center', fontsize=9)
    
    ax.plot([1.5, 4.5], [1.3, 1.3], color='#616161', lw=1.5)
    ax.text(3, 1.0, "Process Flow", ha='center', fontsize=9)
    
    draw_react_badge(2.5, 0.6)
    
    # ============ ORCHESTRATOR ============
    draw_rounded_box(6, -2.5, 12, 0.8, "Master Orchestrator: LangGraph State + Checkpointing", '#F5F5F5', bold=False)

    plt.tight_layout()
    plt.savefig('architecture_diagram.png', bbox_inches='tight', dpi=200, facecolor='white')
    print("Done")

if __name__ == "__main__":
    create_architecture_diagram()
