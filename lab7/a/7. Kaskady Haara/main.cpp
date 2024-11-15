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

## Redukcja błędów wykrycia twarzy

Jak już udało się Tobie uruchomić, to chyba najbardziej uciążliwa część roboty jest zrobiona. Teraz czas na konkrety.

Chcielibyśmy zliczać twarze, ale w sposób jak najbardziej pewny. Połączymy fakt, że w przykładzie korzystamy z kaskad do twarzy oraz oczu.

Zakładamy, że nie będzie sytuacji wyjątkowych (typu pirat, albo Cyklop). Przy takim założeniu, możemy stwierdzić, że na
twarzy powinny być oczy. Możemy to łatwo sprawdzić porównując, czy w wykrytej twarzy (dostajemy prostokąt) jest
też co najmniej jedno oko (wiem, powinny być dwa, ale czasami nie wykryje obu).

W przykładzie mamy to już prawie zrobione - odbywa się wykrywanie oczu w obrębie twarzy (faceROI to wycinek z twarzą - możesz sobie wyświetlić aby sprawdzić).

Zobacz w funkcji detectAndDisplay linię

```c++
std::vector<Rect> faces;
face_cascade.detectMultiScale( frame_gray, faces );
```

To jest moment zastosowania kaskady do wykrycia twarzy na obrazie. zmienna faces przechowuje tablicę prostokątów
które zaznaczają miejsca gdzie wykryto twarz. Jak już pewnie się domyślasz - można to wykorzystać w połączeniu
z poprzednią informacją.

Linie poniższe robią nam robotę wykrycia oczu w obszarze twarzy:

```c++
std::vector<Rect> eyes;
eyes_cascade.detectMultiScale( faceROI, eyes );
```

Mając obie informacje, możemy przerobić tak, aby pokazywało tylko te twarze na których widać oczy. Zrób to.

## Uporządkowanie kodu

Jak już potrafimy wyświetlać tylko twarze które mają oczy, to teraz przygotujemy się do zliczania ile to twarzy
mamy na ekranie.

Przykład ze strony OpenCV ma pewne niedociągnięcia związane z jakością kodu. Na przykład - funkca ```detectAndDisplay```
wykonuje dwie rozłączne czynności za jednym zamachem - detekcję oraz wyświetlanie. Porządniej by było, gdybyśmy to
poprawili i rozdzielili te dwa etapy. Proszę przerobić tak, aby były dwie funkcje:

```c++
std::vector<cv::Rect> detectFaces(/*pomyśl co tu ma być w argumentach*/);
Mat drawFaces(Mat frame, std::vector<cv::Rect> &faces);
```

## Zliczanie twarzy

Zaobserwuj, że mając taką funkcję (detectFaces) możemy skorzystać z rozmiaru zwracanej tablicy i to będzie liczba widocznych twarzy.
Wypisuj na konsolę informację o liczbie twarzy wykrytych na obrazie.

## Sprawdzanie największego tłumu

Dodaj wyszukiwanie maksymalnej liczby, wystarczy std::max. Proszę dokończyć przykład - niech będzie on wyświetlał aktualną liczbę wykrytych twarzy, a w momencie zamknięcia programu niech wypisze maksimum.

## Stan na punkt

Ostatni detal  - bez odbicia lustrzanego trudno jest pokazywać do kamery różne rzeczy. Proszę odbić obraz przed wyświetleniem. Chcielibyśmy, aby
użytkownik widział się jak w lustrze. To będzie ostatnie podstawowe zadanie.

## Zadanie domowe - dostrojenie narzędzia

Czasami zdarza się, że są nakładające się na siebie wykrycia twarzy. Możesz je odfiltrować za pomocą operatora ```&``` zastosowanego na prostokątach.
Jeśli obszary wykrycia są jeden w drógim, to można zrobić coś takiego:

```c++
Rect a(20,20,40,40);
Rect b(30,30,40,40);

if ((a & b) == a) { /* a zawiera w całości b */ }
```

Rozwiń funkcję wykrywania twarzy o pomijanie już wykrytych twarzy. Wystarczy przepisać do nowej tablicy te,
tkóre nie są już wykryte - pętla w pętli wystarczy, ale można też posłużyć się funkcjami z biblioteki algorytmów (dla ambitnych).

###


## Zadanie dodatkowe

Niech wykrywanie twarzy będzie wystawione do zadań asynchronicznych i jak zadanie sie zakończy, to dopiero będzie aktualizowana lista twarzy do wyświetlenia i zliczania. Takie rozwiązanie pozwoli na uzyskanie dużej płynności działania naszego narzędzia.

#endif

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <future>

using namespace std;
using namespace cv;

std::vector<Rect> detectFaces( Mat frame );
Mat drawFaces(Mat frame0, std::vector<cv::Rect> &faces);

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
    int max_detected = 0;
    vector<future<vector<Rect>>> asyncFaceDetect;
//    a2.wait();                     // prints "world!"
//    std::cout << a3.get() << '\n';
    vector<Rect> faces;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        if (asyncFaceDetect.size() == 0) {
            asyncFaceDetect.push_back(std::async(std::launch::async, detectFaces, frame.clone()));
        } else {
            auto &future = asyncFaceDetect[0];
            if (future.wait_for(1ms) == std::future_status::ready) {
                faces = future.get();
                asyncFaceDetect.pop_back();
            }
        }
        if (faces.size() > 0) cout << "Detected " << faces.size() << " faces" << endl;
        max_detected = max(max_detected, (int)faces.size());
        Mat facesImage = drawFaces(frame, faces);
        Mat mirroredFacesImage;
        flip(facesImage, mirroredFacesImage, 1);
        imshow( "Capture - Face detection", mirroredFacesImage );
        if( waitKey(1) == 27 )
        {
            break; // escape
        }
    }
    cout << "MAX DETECTED " << max_detected << endl;
    return 0;
}

std::vector<Rect> detectFaces( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    std::vector<Rect> returnedFaces;
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        if (eyes.size() > 0) {
            returnedFaces.push_back(faces[i]);
        }
    }
    //-- Show what you got
    return returnedFaces;
}


Mat drawFaces(Mat frame0, std::vector<cv::Rect> &faces)
{
    Mat frame = frame0.clone();
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
    }
    return frame;
}