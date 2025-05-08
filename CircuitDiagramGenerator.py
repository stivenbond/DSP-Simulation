import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image
import base64
from matplotlib.path import Path

class CircuitDiagramGenerator:
    """Generate circuit diagrams for op-amp configurations"""

    @staticmethod
    def draw_op_amp(ax, pos=(0, 0), scale=1.0):
        """Draw an op-amp symbol at the given position"""
        # Op-amp triangle
        triangle = patches.Polygon(
            scale * np.array([
                [pos[0], pos[1]],
                [pos[0] - 1, pos[1] + 0.5],
                [pos[0] - 1, pos[1] - 0.5],
            ]),
            closed=True,
            fill=False,
            color='black',
            linewidth=2
        )
        ax.add_patch(triangle)

        # Non-inverting input (+)
        ax.plot(
            [pos[0] - 1, pos[0] - 1.5],
            [pos[1] + 0.25, pos[1] + 0.25],
            'k-', linewidth=2
        )
        ax.text(pos[0] - 1.3, pos[1] + 0.35, '+', fontsize=12)

        # Inverting input (-)
        ax.plot(
            [pos[0] - 1, pos[0] - 1.5],
            [pos[1] - 0.25, pos[1] - 0.25],
            'k-', linewidth=2
        )
        ax.text(pos[0] - 1.3, pos[1] - 0.45, '−', fontsize=12)

        # Output
        ax.plot(
            [pos[0], pos[0] + 0.5],
            [pos[1], pos[1]],
            'k-', linewidth=2
        )

        return ax

    @staticmethod
    def draw_resistor(ax, start, end, label=None, value=None):
        """Draw a resistor between start and end points"""
        # Calculate direction vector
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        unit_dir = direction / length
        perp_dir = np.array([-unit_dir[1], unit_dir[0]]) * 0.05  # Perpendicular direction for zigzag

        # Calculate zigzag points
        num_segments = 7
        segment_length = length / num_segments
        points = []

        points.append(start)
        for i in range(1, num_segments):
            if i % 2 == 1:
                offset = perp_dir
            else:
                offset = -perp_dir

            points.append(np.array(start) + unit_dir * segment_length * i + offset)

        points.append(end)

        # Draw zigzag line
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        ax.plot(x_points, y_points, 'k-', linewidth=2)

        # Add label if provided
        if label:
            midpoint = np.array(start) + direction / 2
            label_offset = perp_dir * 4  # Adjust as needed
            ax.text(midpoint[0] + label_offset[0], midpoint[1] + label_offset[1],
                    f"{label}: {value:.1f} Ω" if value else label,
                    fontsize=10, ha='center')

        return ax

    @staticmethod
    def draw_capacitor(ax, start, end, label=None, value=None):
        """Draw a capacitor between start and end points"""
        # Calculate direction vector
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        unit_dir = direction / length
        perp_dir = np.array([-unit_dir[1], unit_dir[0]]) * 0.1  # Perpendicular for capacitor plates

        # Calculate capacitor plate positions
        plate_dist = 0.05  # Half distance between plates
        plate_length = 0.1  # Length of capacitor plates

        mid = np.array(start) + direction / 2
        plate1_center = mid - unit_dir * plate_dist
        plate2_center = mid + unit_dir * plate_dist

        # Draw wires to capacitor
        ax.plot([start[0], plate1_center[0] - plate_dist],
                [start[1], plate1_center[1] - plate_dist * unit_dir[1]], 'k-', linewidth=2)
        ax.plot([plate2_center[0] + plate_dist, end[0]],
                [plate2_center[1] + plate_dist * unit_dir[1], end[1]], 'k-', linewidth=2)

        # Draw capacitor plates
        ax.plot([plate1_center[0] - perp_dir[0] * plate_length,
                 plate1_center[0] + perp_dir[0] * plate_length],
                [plate1_center[1] - perp_dir[1] * plate_length,
                 plate1_center[1] + perp_dir[1] * plate_length], 'k-', linewidth=2)

        ax.plot([plate2_center[0] - perp_dir[0] * plate_length,
                 plate2_center[0] + perp_dir[0] * plate_length],
                [plate2_center[1] - perp_dir[1] * plate_length,
                 plate2_center[1] + perp_dir[1] * plate_length], 'k-', linewidth=2)

        # Add label if provided
        if label:
            midpoint = np.array(start) + direction / 2
            label_offset = perp_dir * 3  # Adjust as needed
            ax.text(midpoint[0] + label_offset[0], midpoint[1] + label_offset[1],
                    f"{label}: {value:.1e} F" if value else label,
                    fontsize=10, ha='center')

        return ax

    @staticmethod
    def draw_ground(ax, pos):
        """Draw a ground symbol at the given position"""
        # Vertical line
        ax.plot([pos[0], pos[0]], [pos[1], pos[1] - 0.2], 'k-', linewidth=2)

        # Horizontal lines (decreasing in size)
        for i in range(3):
            width = 0.1 - i * 0.02
            ax.plot([pos[0] - width, pos[0] + width],
                    [pos[1] - 0.2 - i * 0.05, pos[1] - 0.2 - i * 0.05],
                    'k-', linewidth=2)

        return ax

    @staticmethod
    def draw_input_source(ax, pos, label="Vin"):
        """Draw an input voltage source symbol"""
        circle = plt.Circle(pos, 0.2, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10)
        return ax

    @staticmethod
    def draw_output_point(ax, pos, label="Vout"):
        """Draw an output measurement point"""
        ax.plot(pos[0], pos[1], 'ko', markersize=4)
        ax.text(pos[0] + 0.1, pos[1] + 0.1, label, fontsize=10)
        return ax

    @staticmethod
    def draw_wire(ax, start, end):
        """Draw a wire between start and end points"""
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
        return ax

    @staticmethod
    def generate_circuit_diagram(circuit_type, parameters):
        """Generate circuit diagram based on circuit type and parameters"""
        fig, ax = plt.subplots(figsize=(6, 4))

        # Common settings
        op_amp_pos = (0, 0)

        # Set axis limits and remove ticks
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 2)
        ax.axis('off')

        # Draw circuit based on type
        if circuit_type == "inverting_amplifier":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Draw input circuit
            input_pos = (-2, -1)
            junction_pos = (-1.5, -0.25)

            # Vin and input resistor
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (junction_pos[0], input_pos[1]))
            CircuitDiagramGenerator.draw_wire(ax, (junction_pos[0], input_pos[1]), junction_pos)
            CircuitDiagramGenerator.draw_resistor(ax, junction_pos, (-1.5, -0.25), "R1", parameters.get("r_input", 1000))

            # Feedback resistor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_resistor(ax, feedback_start, feedback_end, "Rf", parameters.get("r_feedback", 10000))

            # Connect feedback to output
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title
            ax.set_title("Inverting Amplifier Circuit")

        elif circuit_type == "non_inverting_amplifier":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Draw input directly to non-inverting input
            input_pos = (-2, 0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.5, 0.25))

            # Draw ground resistor
            ground_pos = (-1.5, -1)
            CircuitDiagramGenerator.draw_resistor(ax, (-1.5, -0.25), ground_pos, "Rg", parameters.get("r_ground", 1000))
            CircuitDiagramGenerator.draw_ground(ax, ground_pos)

            # Feedback resistor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_resistor(ax, feedback_start, feedback_end, "Rf", parameters.get("r_feedback", 10000))

            # Connect feedback to output
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))

            # Output point
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title
            ax.set_title("Non-Inverting Amplifier Circuit")

        elif circuit_type == "low_pass_filter":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Input and capacitor
            input_pos = (-2, -0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.7, -0.25))
            CircuitDiagramGenerator.draw_capacitor(ax, (-1.7, -0.25), (-1.5, -0.25), "C", 1e-6)

            # Feedback resistor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_resistor(ax, feedback_start, feedback_end, "R", parameters.get("r_feedback", 10000))

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title with cutoff frequency
            cutoff_freq = parameters.get("cutoff_freq", 100)
            ax.set_title(f"Low-Pass Filter (fc = {cutoff_freq} Hz)")

        elif circuit_type == "high_pass_filter":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Input resistor
            input_pos = (-2, -0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.7, -0.25))
            CircuitDiagramGenerator.draw_resistor(ax, (-1.7, -0.25), (-1.5, -0.25), "R", parameters.get("r_input", 10000))

            # Feedback capacitor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_capacitor(ax, feedback_start, feedback_end, "C", 1e-6)

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title with cutoff frequency
            cutoff_freq = parameters.get("cutoff_freq", 100)
            ax.set_title(f"High-Pass Filter (fc = {cutoff_freq} Hz)")

        elif circuit_type == "band_pass_filter":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Input circuit with resistor and capacitor
            input_pos = (-2, -0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.8, -0.25))
            CircuitDiagramGenerator.draw_capacitor(ax, (-1.8, -0.25), (-1.65, -0.25), "C1", 1e-6)
            CircuitDiagramGenerator.draw_resistor(ax, (-1.65, -0.25), (-1.5, -0.25), "R1", parameters.get("r_input", 10000))

            # Feedback network with parallel RC
            feedback_start = (-1.5, -0.25)
            feedback_mid = (-0.5, 0.5)
            feedback_end = (0.5, 0)

            CircuitDiagramGenerator.draw_wire(ax, feedback_start, (-1.5, 0.5))
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.5), feedback_mid)
            CircuitDiagramGenerator.draw_resistor(ax, feedback_mid, (0.5, 0.5), "R2", parameters.get("r_feedback", 10000))
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0.5), feedback_end)
            CircuitDiagramGenerator.draw_capacitor(ax, feedback_mid, (0.5, -0.5), "C2", 1e-6)
            CircuitDiagramGenerator.draw_wire(ax, (0.5, -0.5), feedback_end)

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title with cutoff frequencies
            low_cutoff = parameters.get("cutoff_freq", 100)
            high_cutoff = parameters.get("high_cutoff", 200)
            ax.set_title(f"Band-Pass Filter ({low_cutoff}-{high_cutoff} Hz)")

        elif circuit_type == "integrator":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Input circuit with resistor
            input_pos = (-2, -0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.7, -0.25))
            CircuitDiagramGenerator.draw_resistor(ax, (-1.7, -0.25), (-1.5, -0.25), "R", parameters.get("r_input", 10000))

            # Feedback capacitor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_capacitor(ax, feedback_start, feedback_end, "C", parameters.get("capacitance", 1e-6))

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title 
            ax.set_title("Integrator Circuit")

        elif circuit_type == "differentiator":
            # Draw op-amp
            CircuitDiagramGenerator.draw_op_amp(ax, op_amp_pos)

            # Input circuit with capacitor
            input_pos = (-2, -0.25)
            CircuitDiagramGenerator.draw_input_source(ax, input_pos, "Vin")
            CircuitDiagramGenerator.draw_wire(ax, (input_pos[0] + 0.2, input_pos[1]), (-1.7, -0.25))
            CircuitDiagramGenerator.draw_capacitor(ax, (-1.7, -0.25), (-1.5, -0.25), "C", parameters.get("capacitance", 1e-6))

            # Feedback resistor
            feedback_start = (-1.5, -0.25)
            feedback_end = (0.5, 0)
            CircuitDiagramGenerator.draw_resistor(ax, feedback_start, feedback_end, "R", parameters.get("r_feedback", 10000))

            # Ground the non-inverting input
            CircuitDiagramGenerator.draw_wire(ax, (-1.5, 0.25), (-1.5, 0.75))
            CircuitDiagramGenerator.draw_ground(ax, (-1.5, 0.75))

            # Output point
            CircuitDiagramGenerator.draw_wire(ax, (0.5, 0), (1.5, 0))
            CircuitDiagramGenerator.draw_output_point(ax, (1.5, 0), "Vout")

            # Title
            ax.set_title("Differentiator Circuit")

        else:
            ax.text(0, 0, f"Circuit diagram for {circuit_type} not implemented", ha='center')

        # Convert the plot to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str