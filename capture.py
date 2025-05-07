import pyshark
import threading

# Global variable to store captured packet data
packet_data = []

# Function to capture packets in the background
def capture_packets():
    global packet_data
    try:
        capture = pyshark.LiveCapture(interface='wlo1')  # Adjust 'wlo1' if needed
        
        # Start capturing packets and process them (e.g., capture for 50 seconds)
        capture.sniff(timeout=50)
        
        # Store captured packets in packet_data
        for packet in capture:
            packet_data.append(str(packet))

    except Exception as e:
        print(f"Error while capturing packets: {e}")

# Function to start the packet capture in a background thread
def start_capture():
    capture_thread = threading.Thread(target=capture_packets)
    capture_thread.start()

