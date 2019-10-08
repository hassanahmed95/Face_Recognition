import face_recognition
import cv2
from mtcnn.mtcnn import MTCNN
import imutils


def face_detection(image):
    detector = MTCNN()
    face = detector.detect_faces(image)[0]
    # print(face)

    x, y, width, height = face['box']
    # data = cv2.rectangle(image, (x, y), (x + width+50, y + height+15), (255, 0, 0), 1)
    cropped_face = image[y-20: y + height+20, x - 20: x + width + 20]

    bright_image = increase_brightness(cropped_face, 30)
    bright_image = imutils.resize(bright_image, width=200)
    # cv2.imshow ("bright_image", bright_image)
    dst = cv2.fastNlMeansDenoisingColored(bright_image, None, 8, 8, 7, 12)
    # return bright_image
    return dst


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def face_recog(known_face, unknown_face):
    
    known_encoded = face_recognition.face_encodings(known_face)[0]
    unknown_encoded = face_recognition.face_encodings(unknown_face)[0]
    results = face_recognition.compare_faces([known_encoded], unknown_encoded,tolerance=0.54)
    return results


def main(images):
    image1 = imutils.resize(images[0],width=600)
    face1 = face_detection(image1)

    image2 = imutils.resize(images[1], width=600)
    face2 = face_detection(image2)

    # comparing both faces using face_recognition API. . .

    result = face_recog(face1, face2)
    if result == [True]:
        return ("Both are same persons")
    else:
        return ("Both are different persons")


#
#
# if __name__ == '__main__':
#     main()
