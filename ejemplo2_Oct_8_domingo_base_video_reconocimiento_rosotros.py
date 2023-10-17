
"""
import cv2
import face_recognition


image_to_detect = cv2.imread('images/testing/trump-modi.jpg')


all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')


print('There are {} no of faces in this image'.format(len(all_face_locations)))

for index,current_face_location in enumerate(all_face_locations):
    
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    cv2.imshow("Face no "+str(index+1),current_face_image)

"""
import cv2
import face_recognition

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#Inicializar un array vacío, para localizar la cara
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    
    #looping through the face locations and the face embeddings
    for index,current_face_location in enumerate(all_face_locations):

        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))

        cv2.rectangle(current_frame, ( left_pos,top_pos ), ( right_pos,bottom_pos ), (0,0,255), 2 )

        cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()        





