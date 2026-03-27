import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_box(ax, x, y, w, h, text, facecolor='#f3f4f6', edgecolor='#6b7280', textcolor='#000000', fontsize=10):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                 facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    
    # Text with slight background for readability if needed
    ax.text(x + w/2, y + h/2, text, horizontalalignment='center', verticalalignment='center',
            fontsize=fontsize, color=textcolor, weight='bold')
    return x + w/2, y, x + w/2, y + h

def add_arrow(ax, x1, y1, x2, y2, text=None, text_offset=(0,0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, shrink=0))
    if text:
        ax.text((x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1], text,
                horizontalalignment='center', verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
color_input = '#e1f5fe'
color_input_border = '#3b82f6'
color_module = '#f3f4f6'
color_module_border = '#6b7280'
color_vqvae = '#fef08a'
color_vqvae_border = '#eab308'
color_output = '#dcfce7'
color_output_border = '#ec4899'

# Background Groups
group_core = patches.Rectangle((3.5, 6.5), 9.5, 2.5, linewidth=1.5, edgecolor='black', facecolor='#f9fafb', linestyle='--')
ax.add_patch(group_core)
ax.text(3.7, 8.7, "MambaTalk Core Architecture", fontsize=11, weight='bold', color='#4b5563')

group_vq = patches.Rectangle((2.5, 2.5), 10.5, 2.5, linewidth=1.5, edgecolor='black', facecolor='#fefce8', linestyle='--')
ax.add_patch(group_vq)
ax.text(2.7, 4.7, "Frozen VQ-VAE Decoders", fontsize=11, weight='bold', color='#a16207')

# Layer 1: Inputs (Y=10.5)
ix1, iy1_bot, _, iy1_top = add_box(ax, 1, 10.5, 2, 0.8, "History Motion\n(MHR 323d)", color_input, color_input_border)
ix2, iy2_bot, _, iy2_top = add_box(ax, 4, 10.5, 2, 0.8, "Audio Features\n(Wav2Vec2)", color_input, color_input_border)
ix3, iy3_bot, _, iy3_top = add_box(ax, 7, 10.5, 2, 0.8, "Text Features\n(FastText)", color_input, color_input_border)
ix4, iy4_bot, _, iy4_top = add_box(ax, 10, 10.5, 2, 0.8, "Speaker ID", color_input, color_input_border)

# Layer 2: Fusion (Y=9)
fx1, fy1_bot, _, fy1_top = add_box(ax, 5.5, 9, 3, 0.8, "A-T Softmax Fusion", color_module, color_module_border)
fx2, fy2_bot, _, fy2_top = add_box(ax, 10, 9, 2, 0.8, "ID Embedding", color_module, color_module_border)

add_arrow(ax, ix2, iy2_bot, fx1-0.5, fy1_top)
add_arrow(ax, ix3, iy3_bot, fx1+0.5, fy1_top)
add_arrow(ax, ix4, iy4_bot, fx2, fy2_top)

# Layer 3: Core (Y=7)
cx1, cy1_bot, _, cy1_top = add_box(ax, 4, 7, 3.5, 1.2, "GlobalScan\nSelf-Attn + Mamba", color_module, color_module_border)
cx2, cy2_bot, _, cy2_top = add_box(ax, 9, 7, 3.5, 1.2, "LocalScan\nCross-Attn + Mamba", color_module, color_module_border)

add_arrow(ax, ix1, iy1_bot, cx1, cy1_top)
add_arrow(ax, fx2, fy2_bot, cx1+1.5, cy1_top) # ID to Global
add_arrow(ax, fx1, fy1_bot, cx2-1.5, cy2_top, "Condition") # Fusion to Local
add_arrow(ax, fx2, fy2_bot, cx2+1.5, cy2_top) # ID to Local
add_arrow(ax, cx1+1.75, 7.6, cx2-1.75, 7.6, "Motion Query", (0, 0.3)) # Global to Local

# Layer 4: Classifiers (Y=5.5)
hx1, hy1_bot, _, hy1_top = add_box(ax, 3, 5.5, 2.5, 0.8, "Body+Global Head\n(Linear, 256d)", color_module, color_module_border)
hx2, hy2_bot, _, hy2_top = add_box(ax, 6.5, 5.5, 2.5, 0.8, "Hands Head\n(Linear, 256d)", color_module, color_module_border)
hx3, hy3_bot, _, hy3_top = add_box(ax, 10, 5.5, 2.5, 0.8, "Face Head\n(Linear, 256d)", color_module, color_module_border)

add_arrow(ax, cx2-1.75, cy2_bot, hx1, hy1_top)
add_arrow(ax, cx2-0.5, cy2_bot, hx2, hy2_top)
add_arrow(ax, cx2+0.5, cy2_bot, hx3, hy3_top)

# Layer 5: VQ-VAE (Y=3.5)
vx1, vy1_bot, _, vy1_top = add_box(ax, 3, 3.5, 2.5, 0.8, "Conv1d Decode\n(137d)", color_vqvae, color_vqvae_border)
vx2, vy2_bot, _, vy2_top = add_box(ax, 6.5, 3.5, 2.5, 0.8, "Conv1d Decode\n(108d)", color_vqvae, color_vqvae_border)
vx3, vy3_bot, _, vy3_top = add_box(ax, 10, 3.5, 2.5, 0.8, "Conv1d Decode\n(75d)", color_vqvae, color_vqvae_border)

add_arrow(ax, hx1, hy1_bot, vx1, vy1_top, "Argmax Index", (-0.8, 0))
add_arrow(ax, hx2, hy2_bot, vx2, vy2_top, "Argmax Index", (0.8, 0))
add_arrow(ax, hx3, hy3_bot, vx3, vy3_top, "Argmax Index", (0.8, 0))

# Layer 6: Post Processing (Y=1.5)
px1, py1_bot, _, py1_top = add_box(ax, 3, 1.5, 2.5, 0.8, "Savgol Filter &\nVelocity Integration", color_module, color_module_border)
add_arrow(ax, vx1, vy1_bot, px1, py1_top)

# Layer 7: Output (Y=0.2)
ox1, oy1_bot, _, oy1_top = add_box(ax, 6, 0.2, 4.5, 0.8, "Final Holistic Gesture\n(MHR 323d)", color_output, color_output_border)

# Output arrows
add_arrow(ax, px1, py1_bot, ox1-1.5, oy1_top)
add_arrow(ax, vx2, vy2_bot, ox1, oy1_top)
add_arrow(ax, vx3, vy3_bot, ox1+1.5, oy1_top)

plt.title("MambaTalk_new_512 Architecture", fontsize=16, weight='bold', pad=20)
plt.savefig('/data1/yangzhuowei/MambaTalk_new_512/docs/architecture.png', dpi=300, bbox_inches='tight')
print("Image generated at /data1/yangzhuowei/MambaTalk_new_512/docs/architecture.png")
