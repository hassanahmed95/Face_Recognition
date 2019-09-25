import face_recognition
import cv2
from mtcnn.mtcnn import MTCNN
import imutils

def face_detection(image):
    detector = MTCNN()
    face = detector.detect_faces(image)[0]
    print(face)

    x, y, width, height = face['box']
    # data = cv2.rectangle(image, (x, y), (x + width+50, y + height+15), (255, 0, 0), 1)
    cropped_face = image[y-20: y + height+20, x - 20: x + width + 20]
    # cropped_face = imutils.resize(cropped_face,width=)
    # cropped_face =  cv2.resize(cropped_face,(150,150))
    # cv2.imshow("Cropped", cropped_face)
    # cv2.waitKey()

    # cv2.destroyAllWindows()
    bright_image = increase_brightness(cropped_face,30)
    bright_image =  imutils.resize(bright_image, width=200)
    # cv2.imshow("bright_image", bright_image)
    return bright_image
    # cv2.waitKey()

    # return cropped_face

    # for face in faces:
    #     x, y, width, height = face['box']
    #     # data = cv2.rectangle(image, (x, y), (x + width+50, y + height+15), (255, 0, 0), 1)
    #     cropped_face = image[y: y + height, x-30: x + width+30]
    #     # cropped_face = imutils.resize(cropped_face,width=)
    #     # cropped_face =  cv2.resize(cropped_face,(150,150))
    #     cv2.imshow("Cropped", cropped_face)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()


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
    # known_image = face_recognition.load_image_file(known_face)
    known_image = known_face
    # unknown_image = face_recognition.load_image_file(unknown_face)
    unknown_image = unknown_face

    known_encoded = face_recognition.face_encodings(known_image)[0]
    unknown_encoded = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([known_encoded], unknown_encoded)
    return results


if __name__ == '__main__':
    card_image = cv2.imread("face_images/hassan_pass.jpg")

    face1 = face_detection(card_image)

    selfie_image = "face_images/222.jpg"
    face2 =  cv2.imread(selfie_image)

    result = face_recog(face1,face2)
    print(result)
