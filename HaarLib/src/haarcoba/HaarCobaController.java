/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package haarcoba;

import java.net.URL;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class HaarCobaController implements Initializable {

    @FXML
    private RadioButton haar1;
    @FXML
    private RadioButton haar2;
    @FXML
    private RadioButton haar3;
    @FXML
    private RadioButton haar4;
    @FXML
    private RadioButton haar5;
    @FXML
    private RadioButton haar6;
    @FXML
    private Button btnStartCamera;
    @FXML
    private RadioButton haar7;
    @FXML
    private ImageView frame;
    @FXML
    private ToggleGroup haar;
////

    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;

    private CascadeClassifier haarCascade;
    private int absoluteFaceSize;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        haar1.setText("aGest");
        haar2.setText("closed_frontal_palm");
        haar3.setText("fist");
        haar4.setText("Hand.Cascade.1");
        haar5.setText("hand");
        haar6.setText("palm");
        haar7.setText("downloadandolme12566172332haarcascadehand");
        //
        this.capture = new VideoCapture();
        this.haarCascade = new CascadeClassifier();
        this.absoluteFaceSize = 0;

    }

    @FXML
    private void StartCameraOnAction(ActionEvent event) {
        String s = "E:\\TA\\New Folder\\HaarCoba\\src\\haarcoba\\resource\\";
        RadioButton chk = (RadioButton) haar.getSelectedToggle(); // Cast object to radio button
        s += chk.getText() + ".xml";
        haarCascade.load(s);

        if (!this.cameraActive) {
            // disable setting checkboxes

            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run() {
                        // effectively grab and process a single frame
                        Mat frameMat = grabFrame();
                        // convert and show the frame
                        Image imageToShow = Utils.mat2Image(frameMat);
                        updateImageView(frame, imageToShow);
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33,
                        TimeUnit.MILLISECONDS);

                // update the button content
                this.btnStartCamera.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.btnStartCamera.setText("Start Camera");
            // enable classifiers checkboxes

            // stop the timer
            this.stopAcquisition();
        }
    }

    private Mat grabFrame() {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    // face detection
                    this.detectAndDisplay(frame);
                }

            } catch (Exception e) {
                // log the (full) error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }

        return frame;
    }

    private void detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0) {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0) {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }

        // detect faces
        this.haarCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0
                | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize),
                new Size());

        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(),
                    new Scalar(0, 255, 0), 3);
        }

    }

    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println(
                        "Exception in stopping the frame capture, trying to release the camera now... "
                        + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * Update the {@link ImageView} in the JavaFX main thread
     *
     * @param view the {@link ImageView} to update
     * @param image the {@link Image} to show
     */
    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed() {
        this.stopAcquisition();
    }
}
