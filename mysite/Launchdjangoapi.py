import subprocess
import webbrowser
# INSTALL DJANGO IF THIS DON'T WORK ( PIP INSTALL DJANGO )

def launch_django():
    # STart local server
    subprocess.Popen(['python', 'manage.py', 'runserver'])

    # Open the web page
    webbrowser.open('http://127.0.0.1:8000')  # Remplacez l'URL par celle de votre application si nécessaire

#LAUNCH SITE
launch_django()