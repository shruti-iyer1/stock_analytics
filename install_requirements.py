import subprocess
import sys

def install_requirements(requirements_file):
    """
    Install the packages listed in the given requirements file.
    
    Args:
    requirements_file (str): Path to the requirements.txt file.
    """
    try:
        # Run the pip install command for the requirements file
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"All requirements from {requirements_file} have been installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)

# Example usage
requirements_file = "requirements.txt"  # Path to your requirements file
install_requirements(requirements_file)
