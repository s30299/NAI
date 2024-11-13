#ifdef instrukca_md
# Kaskady Haara w praktyce

Skorzystamy z przykładu https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

## Zadanie dzisiejsze

Przygotujemy na jego bazie projekt, który będzie zliczał, ile maksymalnie jest osób na obrazie.

Chcielibyśmy także, aby wykrywanie było jak najbardziej stabilne.

## Przygotowanie

Na początek ustaw zmienne środowiskowe - OpenCV_DIR oraz PATH.
W katalogu D:\WORK\pantadeusz\...   jest zainstalowana działająca wersja OpenCV, ale zmienna środowiskowa nie
jest ustawiona. Jak już ustawisz, to jako punkt początkowy Twojego projektu proszę połączyć:
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html


### Przypadek z błędem ładowania kaskad

Jest niezerowa szansa, że nie znajdzie kaskad (na przykład na instalacji Linuksowej). Są tu
https://github.com/npinto/opencv/blob/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml
https://github.com/npinto/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
Można je pobrać i zapisać w katalogu projektu, wtedy wystarczy:
 ./cmake-build-debug/07_opencv --eyes_cascade=haarcascade_eye_tree_eyeglasses.xml --face_cascade=haarcascade_frontalface_alt.xml
Możesz też prezrobić przykład tak, aby zawsze ładował kaskady z bieżącego katalogu.

## Przystosowanie do zadania

Jak już udało się Tobie uruchomić, to chyba najbardziej uciążliwa część roboty jest zrobiona. Teraz czas na konkrety.

Chcielibyśmy zliczać twarze, ale w sposób jak najbardziej pewny. Połączymy fakt, że w przykładzie korzystamy z kaskad do twarzy oraz oczu.

Zakładamy, że nie będzie sytuacji wyjątkowych (typu pirat, albo Cyklop). Przy takim założeniu, możemy stwierdzić, że na
twarzy powinny być oczy. Możemy to łatwo sprawdzić porównując, czy w wykrytej twarzy (dostajemy prostokąt) jest
też co najmniej jedno oko (wiem, powinny być dwa, ale czasami nie wykryje obu). Dodaj do przykładu taki warunek i tylko wtedy
rysuj obramowanie twarzy. Najłatwiej zrobić to wykorzystując operator ```&```, na przykład:

```c++
Rect a(20,20,40,40);
Rect b(30,30,40,40);

if ((a & b) == a) { /* a zawiera w całości b */ }
```


#endif

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;
std::vector<cv::Rect> detectFaces(std::vector<Rect> faces);
Mat drawFaces(Mat frame, std::vector<cv::Rect> &faces);
void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
                             "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
                             "{camera|0|Camera device number.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();
    String face_cascade_name = samples::findFile( parser.get<String>("face_cascade") );
    String eyes_cascade_name = samples::findFile( parser.get<String>("eyes_cascade") );
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}


void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces

    std::vector<Rect> faces = detectFaces( frame );
    Mat WhatToDraw = drawFaces(frame,faces);
    face_cascade.detectMultiScale( frame_gray, faces );
    // for ( size_t i = 0; i < faces.size(); i++ )
    // {
    //     // Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    //     // ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
    //     Mat faceROI = frame_gray( faces[i] );
    //     //-- In each face, detect eyes
    //     std::vector<Rect> eyes;
    //     eyes_cascade.detectMultiScale( faceROI, eyes );
    //     for ( size_t j = 0; j < eyes.size(); j++ )
    //     {
    //         Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    //         ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
    //         Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
    //         int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
    //         circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
    //     }
    // }
    //-- Show what you got
    imshow( "Capture - Face detection", WhatToDraw );
}
std::vector<cv::Rect> detectFaces(Mat frame) {
    std::vector<cv::Rect> result;
    Mat frame_gray;
    // cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        result.insert(result.end(), faces.begin(), eyes.end());
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
        }
    }
    return result;

};
Mat drawFaces(Mat frame, std::vector<cv::Rect> &faces) {
    Mat frame_gray;
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
        }
    }
    return  frame;

};
