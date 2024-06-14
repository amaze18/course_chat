import requests
from youtube_transcript_api import YouTubeTranscriptApi

# Function to get video title using YouTube Data API
def get_video_title(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch video details: {response.status_code}")
    data = response.json()
    title = data['items'][0]['snippet']['title']
    return title

# Function to get video transcript using youtube-transcript-api
def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Join the transcript text in a readable format
        transcript = "\n".join([entry['text'] for entry in transcript_list])
        return transcript
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

# Main function
if __name__ == "__main__":
    api_key = "AIzaSyDYaAgEd0WXuSmJcDeIovRn6dRxS5RwtmE"
    video_url = input("Enter the YouTube video URL: ")

    # Extract video ID from URL
    video_id = video_url.split("v=")[-1].split("&")[0]
    
    try:
        # Fetch video title
        title = get_video_title(api_key, video_id)
        print(f"Video Title: {title}\n")
        
        # Fetch video transcript
        transcript = get_video_transcript(video_id)
        print(f"Transcript:\n{transcript}")

        # Store title and transcript in a file
        with open(f"/home/chetan/course_chat/youtube scripts/{title[:10]}.txt", "w", encoding="utf-8") as file:
            file.write(f"Video Title: {title}\n\nTranscript:\n{transcript}")

        print(f"\nTranscript saved to {title[:10]}.txt")
    except Exception as e:
        print(e)
