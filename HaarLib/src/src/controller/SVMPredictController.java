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
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.Mat;
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
public class SVMPredictController implements Initializable {

    @FXML
    private ImageView layarMain;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnUpdateCamera;
    @FXML
    private TextField txtPredictedResult;
    @FXML
    private ComboBox<String> cmbClassifier;
    @FXML
    private BorderPane apHandViewWindow;
    @FXML
    private TextField txtBoxWidth;
    @FXML
    private TextField txtBoxLowerTreshold;
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

    }

    public void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

    /**
     * ######################################################################
     * Method awal untuk membuka kamera dan memanggil method
     * var:
     * boolean cameraActive : titik saat ini
     * VideoCapture capture : titik sebelumnya
     * Runnable frameGrabber :
     * Mat frame :
     * ScheduledExecutorService timer :
     * Button btnStartCamera :
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        treshold = Double.valueOf(txtBoxLowerTreshold.getText());
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
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Mat frame : titik saat ini
     * Image imageToMat : titik sebelumnya
     * Mat hand :
     * Mat tresholded :
     * double x, x_, y, y_:
     * Point p, p_ :
     * List<MatOfPoint> countour :
     * List<MatOfInt4> devOfInt4s :
     * List<MatOfPoint> devOfPoints :
     */
    private void start(Mat frame) {
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        //
        double height_2, width = Double.valueOf(txtBoxWidth.getText());
        height_2 = Preprocessing.getHeight(width, hand.width(), hand.height());
        Mat handView = Preprocessing.getEdgeView(hand.clone(), width, height_2, treshold);
        //
        Mat handPredict = hand.clone();
        getPredictedResult(handPredict, width);
        //
        updateImageView(layarMain, frame);
    }

    /**
     * ######################################################################
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
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

    /**
     * ######################################################################
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * VideoCapture capture : titik saat ini
     * ScheduledExecutorService timer : titik sebelumnya
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
     * update tampilan pada frame utama
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
     *
     * var:
     * VideoCapture capture :
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
     * method browse classifier OnClick
     * memperoleh nilai koordinat layarMain
     * var:
     * TextField txtV : text field menyimpan koordinat X
     * TextField txtS : text field menyimpan koordinat Y
     *
     */
    @FXML
    private void browseClassifierOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Classification Save Lokasi");
        File Path = brows.showDialog(apHandViewWindow.getScene().getWindow());

        File[] files = Path.listFiles();
        ObservableList<String> typeClassifier = FXCollections.observableArrayList();
        for (File file : files) {
            typeClassifier.add(file.getAbsolutePath());
        }
        cmbClassifier.setItems(typeClassifier);
    }

    @FXML
    private void getClassifierOnClick(ActionEvent event) {
        svm = SVM.load(cmbClassifier.getValue());
    }

}
