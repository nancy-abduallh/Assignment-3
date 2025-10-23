import re
import matplotlib.pyplot as plt
import os


def parse_single_log_file(log_file_path):
    """
    Parse a single log file and extract loss and IoU metrics
    """
    with open(log_file_path, 'r') as file:
        content = file.read()

    # Regular expression to extract epoch summaries
    epoch_pattern = r'✅ Epoch (\d+) Summary \(train\):.*?Average Loss: ([\d.]+).*?Average IoU: ([\d.]+)'

    epochs = []
    loss = []
    iou = []

    matches = re.findall(epoch_pattern, content, re.DOTALL)
    for match in matches:
        epoch, loss_val, iou_val = match
        epochs.append(int(epoch))
        loss.append(float(loss_val))
        iou.append(float(iou_val))

    return {
        'epochs': epochs,
        'loss': loss,
        'iou': iou
    }


def parse_both_phases(phase1_file, phase2_file):
    """
    Parse both phase 1 and phase 2 log files
    """
    print(f"Reading Phase 1 from: {phase1_file}")
    phase1_data = parse_single_log_file(phase1_file)

    print(f"Reading Phase 2 from: {phase2_file}")
    phase2_data = parse_single_log_file(phase2_file)

    return {
        'phase1': phase1_data,
        'phase2': phase2_data
    }


def create_plots(data, save_dir):
    """
    Create and save plots for loss and IoU comparison
    """
    # Set up the style
    plt.style.use('default')

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Loss comparison
    ax1.plot(data['phase1']['epochs'], data['phase1']['loss'],
             'b-o', linewidth=2, markersize=6, label='Phase 1', alpha=0.8)
    ax1.plot(data['phase2']['epochs'], data['phase2']['loss'],
             'r-s', linewidth=2, markersize=6, label='Phase 2', alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss: Phase 1 vs Phase 2', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Add some annotations
    if data['phase1']['loss']:
        min_loss_phase1 = min(data['phase1']['loss'])
        ax1.annotate(f'Min Phase 1: {min_loss_phase1:.3f}',
                     xy=(data['phase1']['epochs'][-1], min_loss_phase1),
                     xytext=(10, 10), textcoords='offset points', fontsize=9)

    if data['phase2']['loss']:
        min_loss_phase2 = min(data['phase2']['loss'])
        ax1.annotate(f'Min Phase 2: {min_loss_phase2:.3f}',
                     xy=(data['phase2']['epochs'][-1], min_loss_phase2),
                     xytext=(10, -20), textcoords='offset points', fontsize=9)

    # Plot 2: IoU comparison
    ax2.plot(data['phase1']['epochs'], data['phase1']['iou'],
             'b-o', linewidth=2, markersize=6, label='Phase 1', alpha=0.8)
    ax2.plot(data['phase2']['epochs'], data['phase2']['iou'],
             'r-s', linewidth=2, markersize=6, label='Phase 2', alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU', fontsize=12)
    ax2.set_title('IoU: Phase 1 vs Phase 2', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Add some annotations for IoU
    if data['phase1']['iou']:
        max_iou_phase1 = max(data['phase1']['iou'])
        ax2.annotate(f'Max Phase 1: {max_iou_phase1:.3f}',
                     xy=(data['phase1']['epochs'][data['phase1']['iou'].index(max_iou_phase1)], max_iou_phase1),
                     xytext=(10, 10), textcoords='offset points', fontsize=9)

    if data['phase2']['iou']:
        max_iou_phase2 = max(data['phase2']['iou'])
        ax2.annotate(f'Max Phase 2: {max_iou_phase2:.3f}',
                     xy=(data['phase2']['epochs'][data['phase2']['iou'].index(max_iou_phase2)], max_iou_phase2),
                     xytext=(10, -20), textcoords='offset points', fontsize=9)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    loss_plot_path = os.path.join(save_dir, 'training_comparison.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {loss_plot_path}")

    # Create individual plots for better clarity
    create_individual_plots(data, save_dir)


def create_individual_plots(data, save_dir):
    """
    Create individual plots for loss and IoU
    """
    # Individual Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['phase1']['epochs'], data['phase1']['loss'],
             'b-o', linewidth=2, markersize=6, label='Phase 1', alpha=0.8)
    plt.plot(data['phase2']['epochs'], data['phase2']['loss'],
             'r-s', linewidth=2, markersize=6, label='Phase 2', alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison: Phase 1 vs Phase 2', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_individual_path = os.path.join(save_dir, 'loss_comparison.png')
    plt.savefig(loss_individual_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Individual IoU plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['phase1']['epochs'], data['phase1']['iou'],
             'b-o', linewidth=2, markersize=6, label='Phase 1', alpha=0.8)
    plt.plot(data['phase2']['epochs'], data['phase2']['iou'],
             'r-s', linewidth=2, markersize=6, label='Phase 2', alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title('IoU Comparison: Phase 1 vs Phase 2', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    iou_individual_path = os.path.join(save_dir, 'iou_comparison.png')
    plt.savefig(iou_individual_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Individual plots saved to:\n  {loss_individual_path}\n  {iou_individual_path}")


def main():
    # Define paths for both log files
    base_dir = "outputs/logs/"
    phase1_file = os.path.join(base_dir, "seqtrack-seqtrack_b256.log")
    phase2_file = os.path.join(base_dir, "seqtrack-seqtrack_b256-phase2.log")
    save_dir = base_dir

    # Check if log files exist
    if not os.path.exists(phase1_file):
        print(f"Error: Phase 1 log file not found at {phase1_file}")
        return

    if not os.path.exists(phase2_file):
        print(f"Error: Phase 2 log file not found at {phase2_file}")
        return

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Parse both log files
        print("Parsing log files...")
        data = parse_both_phases(phase1_file, phase2_file)

        # Print some statistics
        if data['phase1']['epochs']:
            print(f"\nPhase 1: Epochs {min(data['phase1']['epochs'])}-{max(data['phase1']['epochs'])}")
            print(f"  Final Loss: {data['phase1']['loss'][-1]:.4f}")
            print(f"  Final IoU: {data['phase1']['iou'][-1]:.4f}")
        else:
            print("\nPhase 1: No data found")

        if data['phase2']['epochs']:
            print(f"\nPhase 2: Epochs {min(data['phase2']['epochs'])}-{max(data['phase2']['epochs'])}")
            print(f"  Final Loss: {data['phase2']['loss'][-1]:.4f}")
            print(f"  Final IoU: {data['phase2']['iou'][-1]:.4f}")
        else:
            print("\nPhase 2: No data found")

        # Check if we have data for both phases
        if not data['phase1']['epochs'] or not data['phase2']['epochs']:
            print("Error: Could not extract data from one or both log files")
            return

        # Create and save plots
        print("\nCreating plots...")
        create_plots(data, save_dir)

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()