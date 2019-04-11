/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package src.controller;

import java.io.File;
import src.utils.Utils;
import java.net.URL;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.videoio.VideoCapture;
import src.utils.DataTrainingPrep;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class LiveTranslationController implements Initializable {

    @FXML
    private TextField txtBoxLowerThreshold;
    @FXML
    private TextField txtBoxMinWidth;
    @FXML
    private Button btnGetClassifier;
    @FXML
    private Button btnBrowseClassifierLocation;
    @FXML
    private ImageView MainFrame;
    @FXML
    private Button btnStartCamera;
    @FXML
    private TextField txtPredictedResult;
    @FXML
    private ComboBox<String> cmbClassifier;
    @FXML
    private BorderPane apHandViewWindow;
//
    private MainAppController mainAppController;
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private SVM svm;
    private double treshold;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        //

        Mat image = new Mat(480, 640, CvType.CV_8UC3, new Scalar(255, 255, 255));
        image = Preprocessing.drawRect(image);
        updateImageView(MainFrame, image);
    }

    /**
     * ######################################################################
     * Method untuk menentukan Main Controller dari tampilan.
     */
    public void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

    /**
     * ######################################################################
     * Method awal untuk memulai membuka kamera dan memulai proses pengambilan gambar.
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        treshold = Double.valueOf(txtBoxLowerThreshold.getText());
        if (!cameraActive) {
            capture.open(0);
            if (this.capture.isOpened()) {
                cameraActive = true;
                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = null;
                        try {
                            frame = grabFrame();
                            start(frame);
                        } catch (Exception e) {
                            System.err.println("startCameraOnClick " + e);
                        }
                    }
                };
                this.timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33,
                        TimeUnit.MILLISECONDS);
                btnStartCamera.setText("stop Camera");
            } else {
                System.err.println("tidak dapat membuka kamera");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.btnStartCamera.setText("Start Camera");
            // stop the timer
            this.stopAcquisition();
        }
    }

    /**
     * ######################################################################
     * Method ini menghentikan thread yang dijalankan oleh method startCameraOnClick.
     *
     */
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
     * ######################################################################
     * Method untuk mengganti gambar dari main frame.
     * var:
     *
     */
    private void updateImageView(ImageView view, Mat image) {
        Image imageToMat;
        imageToMat = Utils.mat2Image(image);
        Utils.onFXThread(view.imageProperty(), imageToMat);
    }

    /**
     * ######################################################################
     * On application close, stop the acquisition from the camera
     * var:
     *
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

    /**
     * ######################################################################
     * Method untuk memperoleh gambar dari kamera.
     *
     */
    private Mat grabFrame() {
        // init everything
        Mat frame = new Mat();
        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);

                }

            } catch (Exception e) {
                // log the error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }
        return frame;
    }

    /**
     * ######################################################################
     * Method onClick untuk mencari lokasi penyimpanan classifier.
     *
     */
    @FXML
    private void browseClassifierLocationOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Open Folder");
        File Path = brows.showDialog(apHandViewWindow.getScene().getWindow());

        File[] files = Path.listFiles();
        ObservableList<String> typeClassifier = FXCollections.observableArrayList();
        for (File file : files) {
            typeClassifier.add(file.getAbsolutePath());
        }
        cmbClassifier.setItems(typeClassifier);
    }

    /**
     * ######################################################################
     * Method onClick untuk me-load classifier
     *
     */
    @FXML
    private void getClassifierOnClick(ActionEvent event) {
        svm = SVM.load(cmbClassifier.getValue());
    }

    /**
     * ######################################################################
     * Method untuk menambahkan ROI pada gambar main frame dan memanggil method lain untuk melakukan pemrosesan gambar dan proses prediksi
     */
    private void start(Mat frame) {
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        //
        double height_2, width = Double.valueOf(txtBoxMinWidth.getText());
        height_2 = Preprocessing.getHeight(width, hand.width(), hand.height());
        //
        Mat handPredict = hand.clone();
        getPredictedResult(handPredict, width);
        //
        updateImageView(MainFrame, frame);
    }

    /**
     * ######################################################################
     * Method untuk memprediksi hasil gambar yang terambil kamera.
     *
     */
    public void getPredictedResult(Mat hand, double width) {
        try {
            Mat dataFile = DataTrainingPrep.getImageEdgeDescriptor(hand.clone(), width, treshold);
            float label = svm.predict(dataFile);
            String result = "";
            if (label == 0) {
                result = "C";
            } else if (label == 1) {
                result = "I";
            } else if (label == 2) {
                result = "L";
            } else if (label == 3) {
                result = "O";
            } else if (label == 4) {
                result = "U";
            } else if (label == 5) {
                result = "V";
            }
            txtPredictedResult.setText(result);
        } catch (Exception e) {
            System.err.println("getPredictedResult " + e);
        }
    }
}
