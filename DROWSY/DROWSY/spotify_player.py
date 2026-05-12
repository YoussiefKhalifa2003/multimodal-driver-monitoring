import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import os
import webbrowser
from dotenv import load_dotenv

load_dotenv()

class SpotifyPlayer:
    def __init__(self):
        self.is_playing = False
        self.sp = None
        self.setup_spotify()
    
    def setup_spotify(self):
        """Setup connection to Spotify API"""
        try:
            # Credentials are loaded from .env — see .env.example at the repo root
            os.environ['SPOTIPY_CLIENT_ID'] = os.getenv('SPOTIFY_CLIENT_ID', '')
            os.environ['SPOTIPY_CLIENT_SECRET'] = os.getenv('SPOTIFY_CLIENT_SECRET', '')
            os.environ['SPOTIPY_REDIRECT_URI'] = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')
            
            scope = "user-read-playback-state,user-modify-playback-state"
            
            # Create Spotify client
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
            print("Spotify connection established")
            
            # Check if there are any active devices
            devices = self.sp.devices()
            if not devices['devices']:
                print("No active Spotify devices found. Please open Spotify on your device.")
                # Open Spotify web player in browser to create an active device
                webbrowser.open('https://open.spotify.com')
                time.sleep(3)  # Give time for Spotify to open
            
        except Exception as e:
            print(f"Error setting up Spotify: {e}")
            print("Please make sure you've set up your Spotify API credentials.")
            self.sp = None
    
    def play_alert_music(self, playlist_uri=None):
        """Play a song from a Spotify playlist as an alert"""
        if not self.sp:
            print("Spotify client not initialized. Cannot play music.")
            return False
            
        try:
            # If no playlist specified, use a default wake-up playlist
            # You can replace this with any Spotify playlist URI
            if not playlist_uri:
                playlist_uri = 'spotify:playlist:37i9dQZF1DX4fpCWaHOned'  # Energetic playlist
            
            # Get available devices
            devices = self.sp.devices()
            
            if not devices['devices']:
                print("No active Spotify devices found. Please open Spotify on your device.")
                return False
                
            # Use the first available device
            device_id = devices['devices'][0]['id']
            
            # Get tracks from playlist
            results = self.sp.playlist_tracks(playlist_uri)
            tracks = results['items']
            
            if not tracks:
                print("No tracks found in playlist")
                return False
                
            # Get a track URI from the playlist
            # For drowsiness alert, use a more energetic track if available
            track_uri = tracks[0]['track']['uri']
            
            # Play the track
            self.sp.start_playback(device_id=device_id, uris=[track_uri])
            self.is_playing = True
            
            print(f"Playing alert music: {tracks[0]['track']['name']} by {tracks[0]['track']['artists'][0]['name']}")
            return True
            
        except Exception as e:
            print(f"Error playing Spotify music: {e}")
            return False
    
    def stop_music(self):
        """Stop playback"""
        if self.sp and self.is_playing:
            try:
                self.sp.pause_playback()
                self.is_playing = False
                print("Spotify playback paused")
                return True
            except Exception as e:
                print(f"Error stopping playback: {e}")
                return False
        return False


# Example usage:
if __name__ == "__main__":
    player = SpotifyPlayer()
    player.play_alert_music()
    time.sleep(10)  # Play for 10 seconds
    player.stop_music()
    print("Spotify player test complete") 