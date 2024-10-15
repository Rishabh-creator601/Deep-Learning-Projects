
import streamlit as st
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,WebRtcMode
import av,time,queue
from main import detect_frame, write_file,read_file,cut_clip
from aiortc.contrib.media import MediaRecorder

import uuid
from pathlib import Path


st.title("Live WEB app ")




RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)










def frame_video(frame: av.VideoFrame):


	img = frame.to_ndarray(format="bgr24")

	img = detect_frame(img)

	return av.VideoFrame.from_ndarray(img, format="bgr24")






if "prefix" not in st.session_state:
	st.session_state["prefix"] = str(uuid.uuid4())
	#print("Got UUID")
prefix = st.session_state["prefix"]
#print(prefix)
in_file = RECORD_DIR / f"{prefix}_input.flv"
	

def recorder_factory() -> MediaRecorder:
	return MediaRecorder(str(in_file),format="flv")


	









web_ctx  = webrtc_streamer(
    key="stream",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    #out_recorder_factory=recorder_factory,
    video_frame_callback=frame_video,
    rtc_configuration = RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
    )






if st.checkbox("Show the detected labels", value=True):
    if web_ctx.state.playing:
        labels_placeholder = st.empty()

        while True:
            cont = read_file("data.txt")
            labels_placeholder.text(cont)









if in_file.exists():

	with in_file.open("rb") as f :


		


		times =  st.selectbox("Select Time Frame (in sec)" ,[2,5,10,15],index=1)

		if st.button("Cut Clip") :

			clip = cut_clip(str(in_file),times)

			file_  =  "./records/edited.mp4"
			clip.write_videofile(file_,codec="libx264")
			
			with open(file_,"rb") as f:
				st.download_button("Download File",f,"file.mp4")



		
		

		

		


