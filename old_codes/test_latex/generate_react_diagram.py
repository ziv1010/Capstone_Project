"""
Generate a professional ReAct framework diagram for research paper
Run this script to create a high-quality PNG diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

# Set up the figure with high DPI for publication quality
fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors - professional academic palette
colors = {
    'thought': '#2196F3',      # Blue
    'action': '#4CAF50',       # Green
    'observe': '#FF9800',      # Orange
    'decision': '#9C27B0',     # Purple
    'light_blue': '#E3F2FD',
    'light_green': '#E8F5E9',
    'light_orange': '#FFF3E0',
    'light_purple': '#F3E5F5'
}

# Helper function to create rounded box with shadow
def create_phase_box(ax, x, y, width, height, color, light_color, title, subtitle, content, content_font='sans-serif'):
    # Shadow
    shadow = FancyBoxatch((x+0.05, y-0.05), width, height, 
                          boxstyle="round,pad=0.1", 
                          facecolor='gray', alpha=0.3, 
                          edgecolor='none', zorder=1)
    ax.add_patch(shadow)
    
    # Main box
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1",
                         facecolor=light_color, 
                         edgecolor=color, 
                         linewidth=2.5, 
                         zorder=2)
    ax.add_patch(box)
    
    # Title
    title_text = ax.text(x + width/2, y + height - 0.3, title,
                        ha='center', va='top', fontsize=11, fontweight='bold',
                        family='sans-serif', zorder=3)
    title_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Subtitle
    ax.text(x + width/2, y + height - 0.6, subtitle,
           ha='center', va='top', fontsize=8, style='italic',
           family='sans-serif', color='gray', zorder=3)
    
    # Content
    if isinstance(content, list):
        y_pos = y + height - 1.0
        for line in content:
            ax.text(x + width/2, y_pos, line,
                   ha='center', va='top', fontsize=7,
                   family=content_font, zorder=3)
            y_pos -= 0.25
    else:
        ax.text(x + width/2, y + height/2 - 0.2, content,
               ha='center', va='center', fontsize=8,
               family=content_font, zorder=3)

# Helper function for curved arrow
def create_curved_arrow(ax, start, end, color, label='', style='solid'):
    arrow = FancyArrowPatch(start, end,
                           connectionstyle="arc3,rad=0.3",
                           arrowstyle='->,head_width=0.4,head_length=0.6',
                           color=color, linewidth=2.5,
                           linestyle=style, zorder=4)
    ax.add_patch(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        text = ax.text(mid_x, mid_y, label,
                      ha='center', va='center', fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='none', alpha=0.9),
                      fontweight='bold', zorder=5)

# Create the 4 phase boxes
# 1. Reasoning Phase (Top)
create_phase_box(ax, 4.5, 7, 3, 2,
                colors['thought'], colors['light_blue'],
                '1. Reasoning Phase', 'Analyze & Plan',
                ['record_thought(',
                 '  thought,',
                 '  what_im_about_to_do',
                 ')'],
                'monospace')

# 2. Acting Phase (Right)
create_phase_box(ax, 8.5, 4, 3, 2,
                colors['action'], colors['light_green'],
                '2. Acting Phase', 'Execute Tools',
                ['• inspect_data_file()',
                 '• run_data_prep_code()',
                 '• save_checkpoint()'],
                'monospace')

# 3. Observation Phase (Bottom)
create_phase_box(ax, 4.5, 1, 3, 2,
                colors['observe'], colors['light_orange'],
                '3. Observation Phase', 'Reflect & Learn',
                ['record_observation(',
                 '  what_happened,',
                 '  what_i_learned,',
                 '  next_step',
                 ')'],
                'monospace')

# 4. Decision Phase (Left)
create_phase_box(ax, 0.5, 4, 3, 2,
                colors['decision'], colors['light_purple'],
                '4. Decision Phase', 'Determine Next Step',
                ['• Continue exploring?',
                 '• Goal achieved?',
                 '• Error to handle?'],
                'sans-serif')

# Center label
ax.text(6, 5, 'ReAct\nCycle',
       ha='center', va='center', fontsize=16, fontweight='bold',
       color='gray', alpha=0.3, family='sans-serif')

# Create arrows
create_curved_arrow(ax, (7.4, 7.8), (8.6, 5.8), colors['thought'], 'Plan')
create_curved_arrow(ax, (9.5, 4.2), (7.4, 2.2), colors['action'], 'Execute')
create_curved_arrow(ax, (4.6, 1.8), (2.4, 4.2), colors['observe'], 'Reflect')
create_curved_arrow(ax, (1.5, 5.8), (4.6, 7.8), colors['decision'], 'Iterate', 'dashed')

# Exit arrow from decision
exit_arrow = FancyArrowPatch((0.5, 5), (0, 5),
                            arrowstyle='->,head_width=0.4,head_length=0.6',
                            color=colors['decision'], linewidth=2.5,
                            linestyle='dashed', zorder=4)
ax.add_patch(exit_arrow)
ax.text(-0.2, 5.3, 'Complete', ha='right', va='bottom', fontsize=8,
       style='italic', color=colors['decision'])

# Example trace box (bottom right)
example_box = FancyBboxPatch((7.5, 0.2), 4.3, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#F5F5F5',
                            edgecolor='#CCCCCC',
                            linewidth=1.5, zorder=2)
ax.add_patch(example_box)

ax.text(9.65, 1.7, 'Example: Stage 3B Data Prep',
       ha='center', va='top', fontsize=9, fontweight='bold',
       family='sans-serif')

example_lines = [
    ('●', colors['thought'], 'Need to load Stage 3 plan'),
    ('●', colors['action'], 'load_stage3_plan_for_prep()'),
    ('●', colors['observe'], 'Plan loaded: 5 files to merge'),
    ('●', colors['decision'], 'Next: Inspect join columns'),
    ('●', colors['thought'], 'Verify join keys exist'),
    ('●', colors['action'], 'inspect_data_file(...)'),
    ('●', colors['observe'], 'Keys confirmed'),
    ('●', colors['decision'], 'Next: Execute merge...')
]

y_pos = 1.4
for marker, color, text in example_lines:
    ax.text(7.7, y_pos, marker, ha='left', va='top', fontsize=8,
           color=color, fontweight='bold')
    ax.text(8.0, y_pos, text, ha='left', va='top', fontsize=7,
           family='monospace' if '()' in text else 'sans-serif')
    y_pos -= 0.16

# Applied in stages box (bottom left)
stages_box = FancyBboxPatch((0.2, 0.2), 3.8, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='white',
                           edgecolor='#CCCCCC',
                           linewidth=1.5, zorder=2)
ax.add_patch(stages_box)

ax.text(2.1, 0.6, 'Applied in Stages 2, 3B, and 5\nfor strategic exploration\nand error recovery',
       ha='center', va='center', fontsize=8, style='italic',
       family='sans-serif', color='#555555')

# Add title
fig.suptitle('ReAct Framework: Reasoning and Acting in Multi-Agent Pipeline',
            fontsize=14, fontweight='bold', y=0.98, family='sans-serif')

plt.tight_layout()

# Save with high quality
output_path = '/scratch/ziv_baretto/llmserve/final_code/test_latex/react_diagram_professional.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✅ Professional ReAct diagram saved to: {output_path}")
print(f"📊 Ready to include in your research paper!")
print(f"\nTo add to LaTeX report:")
print(f"\\includegraphics[width=0.9\\textwidth]{{react_diagram_professional.png}}")

plt.close()
