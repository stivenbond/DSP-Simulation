import flet as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import signal
import io
from PIL import Image
import base64

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

class SignalGenerator:
    """Generate different types of input signals"""

    @staticmethod
    def sine_wave(frequency, amplitude, phase, duration, sampling_rate):
        """Generate a sine wave"""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return t, amplitude * np.sin(2 * np.pi * frequency * t + phase)

    @staticmethod
    def square_wave(frequency, amplitude, duty_cycle, duration, sampling_rate):
        """Generate a square wave"""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return t, amplitude * signal.square(2 * np.pi * frequency * t, duty=duty_cycle)

    @staticmethod
    def triangle_wave(frequency, amplitude, duration, sampling_rate):
        """Generate a triangle wave"""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return t, amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)

    @staticmethod
    def sawtooth_wave(frequency, amplitude, duration, sampling_rate):
        """Generate a sawtooth wave"""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return t, amplitude * signal.sawtooth(2 * np.pi * frequency * t)

    @staticmethod
    def noise(amplitude, duration, sampling_rate):
        """Generate white noise"""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return t, amplitude * np.random.normal(0, 1, len(t))


class OpAmpCircuits:
    """Collection of op-amp circuit models"""

    @staticmethod
    def inverting_amplifier(input_signal, r_feedback, r_input):
        """
        Inverting amplifier: Vout = -(Rf/Rin) * Vin
        Args:
            input_signal: Input voltage signal
            r_feedback: Feedback resistor value (ohms)
            r_input: Input resistor value (ohms)
        """
        gain = -r_feedback / r_input
        return gain * input_signal

    @staticmethod
    def non_inverting_amplifier(input_signal, r_feedback, r_ground):
        """
        Non-inverting amplifier: Vout = (1 + Rf/Rg) * Vin
        Args:
            input_signal: Input voltage signal
            r_feedback: Feedback resistor value (ohms)
            r_ground: Ground resistor value (ohms)
        """
        gain = 1 + (r_feedback / r_ground)
        return gain * input_signal

    @staticmethod
    def low_pass_filter(input_signal, t, cutoff_freq, order=1):
        """
        Low pass filter implementation
        Args:
            input_signal: Input voltage signal
            t: Time values corresponding to input_signal
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order
        """
        # Create Butterworth filter
        b, a = signal.butter(order, cutoff_freq, 'low', analog=False, fs=1/(t[1]-t[0]))
        # Apply filter
        return signal.lfilter(b, a, input_signal)

    @staticmethod
    def high_pass_filter(input_signal, t, cutoff_freq, order=1):
        """
        High pass filter implementation
        Args:
            input_signal: Input voltage signal
            t: Time values corresponding to input_signal
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order
        """
        # Create Butterworth filter
        b, a = signal.butter(order, cutoff_freq, 'high', analog=False, fs=1/(t[1]-t[0]))
        # Apply filter
        return signal.lfilter(b, a, input_signal)

    @staticmethod
    def band_pass_filter(input_signal, t, low_cutoff, high_cutoff, order=1):
        """
        Band pass filter implementation
        Args:
            input_signal: Input voltage signal
            t: Time values corresponding to input_signal
            low_cutoff: Lower cutoff frequency in Hz
            high_cutoff: Higher cutoff frequency in Hz
            order: Filter order
        """
        # Create Butterworth filter
        b, a = signal.butter(order, [low_cutoff, high_cutoff], 'band', analog=False, fs=1/(t[1]-t[0]))
        # Apply filter
        return signal.lfilter(b, a, input_signal)

    @staticmethod
    def integrator(input_signal, t, r_input, capacitance):
        """
        Op-amp integrator: Vout = -1/(R*C) * ∫Vin dt
        Args:
            input_signal: Input voltage signal
            t: Time values
            r_input: Input resistor value (ohms)
            capacitance: Feedback capacitor value (farads)
        """
        dt = t[1] - t[0]  # Time step
        rc = r_input * capacitance
        result = np.zeros_like(input_signal)

        # Simple numerical integration
        integral = 0
        for i in range(len(input_signal)):
            integral += input_signal[i] * dt
            result[i] = -integral / rc

        return result

    @staticmethod
    def differentiator(input_signal, t, r_feedback, capacitance):
        """
        Op-amp differentiator: Vout = -R*C * d(Vin)/dt
        Args:
            input_signal: Input voltage signal
            t: Time values
            r_feedback: Feedback resistor value (ohms)
            capacitance: Input capacitor value (farads)
        """
        # Calculate the derivative using numpy's diff function
        derivative = np.diff(input_signal) / np.diff(t)
        # Pad to match original length
        derivative = np.append(derivative, derivative[-1])

        return -r_feedback * capacitance * derivative

    @staticmethod
    def summing_amplifier(input_signals, resistors, r_feedback):
        """
        Summing amplifier: Vout = -Rf * (V1/R1 + V2/R2 + ... + Vn/Rn)
        Args:
            input_signals: List of input voltage signals
            resistors: List of input resistor values (ohms)
            r_feedback: Feedback resistor value (ohms)
        """
        result = np.zeros_like(input_signals[0])
        for i in range(len(input_signals)):
            result += input_signals[i] / resistors[i]
        return -r_feedback * result


class SignalPlotter:
    """Utility for plotting signals in Flet"""

    @staticmethod
    def plot_signal(t, signal_data, title="Signal", x_label="Time (s)", y_label="Amplitude (V)"):
        """Plot signal and return base64 encoded png"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, signal_data)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        plt.close(fig)

        # Convert to base64 for display in Flet
        buf.seek(0)
        img = Image.open(buf)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    @staticmethod
    def plot_comparison(t, input_signal, output_signal, title="Signal Comparison"):
        """Plot input and output signals and return base64 encoded png"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, input_signal, label="Input Signal")
        ax.plot(t, output_signal, label="Output Signal")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.grid(True)
        ax.legend()

        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        plt.close(fig)

        # Convert to base64 for display in Flet
        buf.seek(0)
        img = Image.open(buf)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str


def main(page: ft.Page):
    page.title = "Op-Amp Signal Processing Simulator"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.scroll = "auto"

    # Current state
    current_state = {
        "signal_type": "sine",
        "circuit_type": "inverting_amplifier",
        "sampling_rate": 1000,  # Hz
        "duration": 1.0,  # seconds
        "input_frequency": 10.0,  # Hz
        "input_amplitude": 1.0,  # Volts
        "input_phase": 0.0,  # radians
        "input_duty_cycle": 0.5,  # for square wave
        "r_feedback": 10000.0,  # ohms (10k)
        "r_input": 1000.0,  # ohms (1k)
        "r_ground": 1000.0,  # ohms (1k)
        "capacitance": 1e-6,  # farads (1μF)
        "cutoff_freq": 100.0,  # Hz
        "high_cutoff": 200.0,  # Hz
        "filter_order": 2,
    }

    # Display elements
    title = ft.Text("Op-Amp Signal Processing Simulator", size=32, weight=ft.FontWeight.BOLD)
    subtitle = ft.Text("Simulate and visualize op-amp circuit responses to different input signals", size=16)

    # Input signal controls
    signal_type_dropdown = ft.Dropdown(
        label="Input Signal Type",
        options=[
            ft.dropdown.Option("sine", "Sine Wave"),
            ft.dropdown.Option("square", "Square Wave"),
            ft.dropdown.Option("triangle", "Triangle Wave"),
            ft.dropdown.Option("sawtooth", "Sawtooth Wave"),
            ft.dropdown.Option("noise", "White Noise"),
        ],
        value=current_state["signal_type"],
        width=200,
    )

    frequency_slider = ft.Slider(
        min=1.0,
        max=100.0,
        divisions=99,
        label="Frequency: {value} Hz",
        value=current_state["input_frequency"],
        width=400,
    )

    amplitude_slider = ft.Slider(
        min=0.1,
        max=5.0,
        divisions=49,
        label="Amplitude: {value} V",
        value=current_state["input_amplitude"],
        width=400,
    )

    phase_slider = ft.Slider(
        min=0.0,
        max=2*np.pi,
        divisions=100,
        label="Phase: {value} rad",
        value=current_state["input_phase"],
        width=400,
    )

    duty_cycle_slider = ft.Slider(
        min=0.1,
        max=0.9,
        divisions=8,
        label="Duty Cycle: {value}",
        value=current_state["input_duty_cycle"],
        width=400,
        visible=current_state["signal_type"] == "square",
    )

    duration_slider = ft.Slider(
        min=0.1,
        max=5.0,
        divisions=49,
        label="Duration: {value} s",
        value=current_state["duration"],
        width=400,
    )

    # Op-amp circuit controls
    circuit_type_dropdown = ft.Dropdown(
        label="Circuit Type",
        options=[
            ft.dropdown.Option("inverting_amplifier", "Inverting Amplifier"),
            ft.dropdown.Option("non_inverting_amplifier", "Non-Inverting Amplifier"),
            ft.dropdown.Option("low_pass_filter", "Low-Pass Filter"),
            ft.dropdown.Option("high_pass_filter", "High-Pass Filter"),
            ft.dropdown.Option("band_pass_filter", "Band-Pass Filter"),
            ft.dropdown.Option("integrator", "Integrator"),
            ft.dropdown.Option("differentiator", "Differentiator"),
        ],
        value=current_state["circuit_type"],
        width=250,
    )

    # Circuit parameter controls (with intelligent visibility)
    r_feedback_slider = ft.Slider(
        min=1000.0,
        max=100000.0,
        divisions=99,
        label="Feedback Resistor: {value} Ω",
        value=current_state["r_feedback"],
        width=400,
    )

    r_input_slider = ft.Slider(
        min=100.0,
        max=10000.0,
        divisions=99,
        label="Input Resistor: {value} Ω",
        value=current_state["r_input"],
        width=400,
    )

    r_ground_slider = ft.Slider(
        min=100.0,
        max=10000.0,
        divisions=99,
        label="Ground Resistor: {value} Ω",
        value=current_state["r_ground"],
        width=400,
        visible=current_state["circuit_type"] == "non_inverting_amplifier",
    )

    capacitance_slider = ft.Slider(
        min=1e-8,
        max=1e-5,
        divisions=100,
        label="Capacitance: {value:.2e} F",
        value=current_state["capacitance"],
        width=400,
        visible=current_state["circuit_type"] in ["integrator", "differentiator"],
    )

    cutoff_freq_slider = ft.Slider(
        min=1.0,
        max=500.0,
        divisions=499,
        label="Cutoff Frequency: {value} Hz",
        value=current_state["cutoff_freq"],
        width=400,
        visible=current_state["circuit_type"] in ["low_pass_filter", "high_pass_filter"],
    )

    high_cutoff_slider = ft.Slider(
        min=1.0,
        max=500.0,
        divisions=499,
        label="High Cutoff Frequency: {value} Hz",
        value=current_state["high_cutoff"],
        width=400,
        visible=current_state["circuit_type"] == "band_pass_filter",
    )

    filter_order_slider = ft.Slider(
        min=1,
        max=8,
        divisions=7,
        label="Filter Order: {value}",
        value=current_state["filter_order"],
        width=400,
        visible=current_state["circuit_type"] in ["low_pass_filter", "high_pass_filter", "band_pass_filter"],
    )

    # Simulation and visualization
    simulate_button = ft.ElevatedButton("Run Simulation", width=200)
    input_plot = ft.Image(src_base64=None , width=600, height=250)
    output_plot = ft.Image(src_base64=None, width=600, height=250)
    comparison_plot = ft.Image(src_base64=None, width=600, height=300)

    circuit_diagram = ft.Image(src_base64="", width=400, height=300)

    # Status display
    status_text = ft.Text("Ready to simulate", size=16)

    # Update UI based on selections
    def update_ui_visibility():
        # Update signal controls
        duty_cycle_slider.visible = signal_type_dropdown.value == "square"
        phase_slider.visible = signal_type_dropdown.value == "sine"

        # Update circuit controls based on selected circuit
        circuit_type = circuit_type_dropdown.value

        r_feedback_slider.visible = circuit_type in ["inverting_amplifier", "non_inverting_amplifier",
                                                     "differentiator", "summing_amplifier"]

        r_input_slider.visible = circuit_type in ["inverting_amplifier", "integrator", "summing_amplifier"]

        r_ground_slider.visible = circuit_type == "non_inverting_amplifier"

        capacitance_slider.visible = circuit_type in ["integrator", "differentiator"]

        cutoff_freq_slider.visible = circuit_type in ["low_pass_filter", "high_pass_filter"]

        high_cutoff_slider.visible = circuit_type == "band_pass_filter"

        filter_order_slider.visible = circuit_type in ["low_pass_filter", "high_pass_filter", "band_pass_filter"]

        # Update circuit diagram
        update_circuit_diagram()

        page.update()

    def update_circuit_diagram():
        # Generate parameters dictionary for the circuit diagram
        params = {
            "r_feedback": r_feedback_slider.value,
            "r_input": r_input_slider.value,
            "r_ground": r_ground_slider.value,
            "capacitance": capacitance_slider.value,
            "cutoff_freq": cutoff_freq_slider.value,
            "high_cutoff": high_cutoff_slider.value,
            # Add other slider/parameter values as needed
        }

        # Get the selected circuit type
        circuit_type = circuit_type_dropdown.value

        # Generate the circuit diagram image
        diagram_base64 = CircuitDiagramGenerator.generate_circuit_diagram(
        circuit_type, params
        )

        # Update circuit diagram display
        circuit_diagram.src_base64 = diagram_base64

        # Generate and process signals
    def run_simulation(e):
        try:
            status_text.value = "Simulating..."
            page.update()

            # Update current state from UI controls
            current_state["signal_type"] = signal_type_dropdown.value
            current_state["circuit_type"] = circuit_type_dropdown.value
            current_state["input_frequency"] = frequency_slider.value
            current_state["input_amplitude"] = amplitude_slider.value
            current_state["input_phase"] = phase_slider.value
            current_state["input_duty_cycle"] = duty_cycle_slider.value
            current_state["duration"] = duration_slider.value
            current_state["r_feedback"] = r_feedback_slider.value
            current_state["r_input"] = r_input_slider.value
            current_state["r_ground"] = r_ground_slider.value
            current_state["capacitance"] = capacitance_slider.value
            current_state["cutoff_freq"] = cutoff_freq_slider.value
            current_state["high_cutoff"] = high_cutoff_slider.value
            current_state["filter_order"] = int(filter_order_slider.value)

            # Generate input signal
            signal_gen = SignalGenerator()
            t = None
            input_signal = None

            if current_state["signal_type"] == "sine":
                t, input_signal = signal_gen.sine_wave(
                    current_state["input_frequency"],
                    current_state["input_amplitude"],
                    current_state["input_phase"],
                    current_state["duration"],
                    current_state["sampling_rate"]
                )
            elif current_state["signal_type"] == "square":
                t, input_signal = signal_gen.square_wave(
                    current_state["input_frequency"],
                    current_state["input_amplitude"],
                    current_state["input_duty_cycle"],
                    current_state["duration"],
                    current_state["sampling_rate"]
                )
            elif current_state["signal_type"] == "triangle":
                t, input_signal = signal_gen.triangle_wave(
                    current_state["input_frequency"],
                    current_state["input_amplitude"],
                    current_state["duration"],
                    current_state["sampling_rate"]
                )
            elif current_state["signal_type"] == "sawtooth":
                t, input_signal = signal_gen.sawtooth_wave(
                    current_state["input_frequency"],
                    current_state["input_amplitude"],
                    current_state["duration"],
                    current_state["sampling_rate"]
                )
            elif current_state["signal_type"] == "noise":
                t, input_signal = signal_gen.noise(
                    current_state["input_amplitude"],
                    current_state["duration"],
                    current_state["sampling_rate"]
                )

            # Process signal through selected circuit
            op_amp = OpAmpCircuits()
            output_signal = None

            circuit_type = current_state["circuit_type"]
            if circuit_type == "inverting_amplifier":
                output_signal = op_amp.inverting_amplifier(
                    input_signal,
                    current_state["r_feedback"],
                    current_state["r_input"]
                )
            elif circuit_type == "non_inverting_amplifier":
                output_signal = op_amp.non_inverting_amplifier(
                    input_signal,
                    current_state["r_feedback"],
                    current_state["r_ground"]
                )
            elif circuit_type == "low_pass_filter":
                output_signal = op_amp.low_pass_filter(
                    input_signal,
                    t,
                    current_state["cutoff_freq"],
                    current_state["filter_order"]
                )
            elif circuit_type == "high_pass_filter":
                output_signal = op_amp.high_pass_filter(
                    input_signal,
                    t,
                    current_state["cutoff_freq"],
                    current_state["filter_order"]
                )
            elif circuit_type == "band_pass_filter":
                output_signal = op_amp.band_pass_filter(
                    input_signal,
                    t,
                    current_state["cutoff_freq"],
                    current_state["high_cutoff"],
                    current_state["filter_order"]
                )
            elif circuit_type == "integrator":
                output_signal = op_amp.integrator(
                    input_signal,
                    t,
                    current_state["r_input"],
                    current_state["capacitance"]
                )
            elif circuit_type == "differentiator":
                output_signal = op_amp.differentiator(
                    input_signal,
                    t,
                    current_state["r_feedback"],
                    current_state["capacitance"]
                )

            # Update plots
            plotter = SignalPlotter()
            input_plot.src_base64 = plotter.plot_signal(t, input_signal, "Input Signal")
            output_plot.src_base64 = plotter.plot_signal(t, output_signal, "Output Signal")
            comparison_plot.src_base64 = plotter.plot_comparison(t, input_signal, output_signal)

            status_text.value = "Simulation complete"
            page.update()

        except Exception as ex:
            status_text.value = f"Error: {str(ex)}"
            page.update()

    # Set up event handlers
    signal_type_dropdown.on_change = lambda e: update_ui_visibility()
    circuit_type_dropdown.on_change = lambda e: update_ui_visibility()
    simulate_button.on_click = run_simulation

    # Create layout
    page.add(
        title,
        subtitle,
        ft.Divider(),

        # Signal controls section
        ft.Text("Input Signal Configuration", size=20, weight=ft.FontWeight.BOLD),
        ft.Row([
            signal_type_dropdown,
        ]),
        ft.Column([
            frequency_slider,
            amplitude_slider,
            phase_slider,
            duty_cycle_slider,
            duration_slider,
        ]),
        ft.Divider(),

        # Circuit controls section
        ft.Text("Op-Amp Circuit Configuration", size=20, weight=ft.FontWeight.BOLD),
        ft.Row([
            circuit_type_dropdown,
        ]),
        ft.Column([
            r_feedback_slider,
            r_input_slider,
            r_ground_slider,
            capacitance_slider,
            cutoff_freq_slider,
            high_cutoff_slider,
            filter_order_slider,
        ]),
        ft.Divider(),

        # Simulation controls and results
        ft.Row([
            simulate_button,
            status_text,
        ]),
        ft.Row([
            ft.Column([
                ft.Text("Input Signal", size=16, weight=ft.FontWeight.BOLD),
                input_plot,
                ft.Text("Output Signal", size=16, weight=ft.FontWeight.BOLD),
                output_plot,
                ft.Text("Signal Comparison", size=16, weight=ft.FontWeight.BOLD),
                comparison_plot,
            ]),
        ]),
    )

    # Initialize UI visibility
    update_ui_visibility()


if __name__ == "__main__":
    ft.app(target=main)