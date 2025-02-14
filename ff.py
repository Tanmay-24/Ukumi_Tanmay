import json
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import subprocess
import os

class VideoOverlay:
    def __init__(self, video_path: str, emphasis_json_path: str):
        self.video_path = video_path
        with open(emphasis_json_path, 'r') as f:
            self.emphasis_data = json.load(f)

    def process_with_moviepy(self, output_path: str):
        """
        Process video using MoviePy - more flexible for styling but slower
        """
        # Load the video
        video = VideoFileClip(self.video_path)
        
        # Get video dimensions
        w, h = video.size
        
        # Create text clips for each emphasis
        text_clips = []
        for emphasis in self.emphasis_data["emphases"]:
            # Create text clip
            txt_clip = TextClip(
                emphasis["overlay_text"],
                fontsize=40,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2
            )
            
            # Set position to center
            txt_clip = txt_clip.set_position(('center', 'center'))
            
            # Set duration and start time
            txt_clip = txt_clip.set_start(emphasis["insert_start"]).set_duration(
                emphasis["insert_end"] - emphasis["insert_start"]
            )
            
            text_clips.append(txt_clip)
        
        # Combine video with all text clips
        final_video = CompositeVideoClip([video] + text_clips)
        
        # Write output
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac'
        )
        
        # Close clips
        video.close()
        final_video.close()
        for clip in text_clips:
            clip.close()

    def generate_ffmpeg_filter(self):
        """
        Generate FFmpeg filter complex string for overlays
        """
        # Start with base filter
        filter_parts = []
        
        for i, emphasis in enumerate(self.emphasis_data["emphases"]):
            # Calculate duration
            duration = emphasis["insert_end"] - emphasis["insert_start"]
            
            # Create drawtext filter
            filter_parts.append(
                f"drawtext=text='{emphasis['overlay_text']}':"
                f"fontsize=40:"
                f"fontcolor=white:"
                f"fontfile=/path/to/font.ttf:"  # Replace with actual font path
                f"x=(w-text_w)/2:"  # Center horizontally
                f"y=(h-text_h)/2:"  # Center vertically
                f"enable='between(t,{emphasis['insert_start']},{emphasis['insert_end']})':"
                f"borderw=2:"
                f"bordercolor=black"
            )
        
        return ','.join(filter_parts)

    def process_with_ffmpeg(self, output_path: str):
        """
        Process video using FFmpeg - faster but less flexible for styling
        """
        filter_complex = self.generate_ffmpeg_filter()
        
        command = [
            'ffmpeg',
            '-i', self.video_path,
            '-vf', filter_complex,
            '-c:a', 'copy',
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running FFmpeg: {e}")
            raise

def main():
    # Example usage
    video_path = "/home/tanmay/Desktop/Ukumi_Tanmay/LTT.mp4"
    emphasis_json_path = "/home/tanmay/Desktop/Ukumi_Tanmay/output/emphasis.json"
    output_path = "/home/tanmay/Desktop/Ukumi_Tanmay/output/output.mp4"
    
    processor = VideoOverlay(video_path, emphasis_json_path)
    
    # Choose one of these methods:
    
    # Method 1: MoviePy
    # processor.process_with_moviepy(output_path)
    
    # Method 2: FFmpeg
    processor.process_with_ffmpeg(output_path)

if __name__ == "__main__":
    main()