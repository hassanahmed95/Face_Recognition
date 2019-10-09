# the boto3 is the python package which will be used to interact python application
# with the AWS service. boto3 is AWS software development kit. .  .
import boto3


# the credientials and config file of AWS are placed at the following
# path .aws/credientials and .aws/config at the root folder of the
# system . ..
# hy #hy
if __name__ == "__main__":
    sourceFile = "face_images/aw.jpg"
    targetFile = "face_images/aw2.jpeg"
    # rekognition is the name of the service to compare both faces
    client = boto3.client('rekognition')

    imageSource = open(sourceFile, 'rb')
    imageTarget = open(targetFile, 'rb')

    response = client.compare_faces(SimilarityThreshold=90,
                                    SourceImage={'Bytes': imageSource.read()},
                                    TargetImage={'Bytes': imageTarget.read()})

    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        confidence = str(faceMatch['Face']['Confidence'])
        print('The face at ' +
              str(position['Left']) + ' ' +
              str(position['Top']) +
              ' matches with ' + confidence + '% confidence')

    imageSource.close()
    imageTarget.close()
