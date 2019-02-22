/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package src.controller;

import java.io.File;
import src.Utils;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;
import src.utils.DataTrainingPrep;
import static src.utils.DataTrainingPrep.getImageEdgeDescriptor;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class HandViewController implements Initializable {

    @FXML
    private ImageView layarBW;
    @FXML
    private ImageView layarEdge;
    @FXML
    private ImageView layarMain;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnUpdateCamera;
    @FXML
    private TextField txtMainFramePoint;
    @FXML
    private TextField txtPredictedResult;
    @FXML
    private ComboBox<String> cmbClassifier;
    @FXML
    private AnchorPane apHandViewWindow;
//
    private MainAppController mainAppController;
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private SVM svm;
    @FXML
    private ImageView layarHand;
    @FXML
    private ComboBox<?> cmbDescriptor;
    @FXML
    private TextField txtBoxWidth;

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
                            System.out.println("startCameraOnClick " + e);
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
        double width_2, height_2, width = Double.valueOf(txtBoxWidth.getText());
        width_2 = hand.width() * (width / hand.width());
        height_2 = hand.height() * (width / hand.width());
        Mat handView = Preprocessing.getEdge_2(hand.clone(), width_2, height_2);
        //
        Mat handPredict = hand.clone();
        getPredictedResult(handPredict, width_2);
        //
        updateImageView(layarMain, frame);
        updateImageView(layarBW, hand);
        updateImageView(layarEdge, handView);
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
            Mat dataFile = DataTrainingPrep.getImageEdgeDescriptor(hand.clone(), width);
            float label = svm.predict(dataFile);
            txtPredictedResult.setText(String.valueOf(label));
            System.out.println(dataFile.rows() + " " + dataFile.cols());
            System.out.println(hand.rows() + " " + hand.cols());
        } catch (Exception e) {
            System.out.println("getPredictedResult " + e);
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
     * method layarBW OnClick
     * method untuk menukar image pada layarBW ke Main
     * var:
     * Image Mn : titik saat ini
     * ImageView layarBW : titik sebelumnya
     * ImageView layarMain
     */
    @FXML
    private void bwToMn(MouseEvent event) {
        Image Mn = layarBW.getImage();
        layarBW.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    /**
     * ######################################################################
     * method layarEdge OnClick
     * method untuk menukar image pada layarEdge ke Main
     * var:
     * Image Mn : titik saat ini
     * ImageView layarEdge : titik sebelumnya
     * ImageView layarMain
     */
    @FXML
    private void edgeToMn(MouseEvent event) {
        Image Mn = layarEdge.getImage();
        layarEdge.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    /**
     * ######################################################################
     * method layarMain OnClick
     * memperoleh nilai koordinat layarMain
     * var:
     * TextField txtV : text field menyimpan koordinat X
     * TextField txtS : text field menyimpan koordinat Y
     *
     */
    @FXML
    private void getPoint(MouseEvent event) {
        txtMainFramePoint.setText("(" + String.valueOf(event.getX()) + ", " + String.valueOf(event.getY()) + ")");
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

//        svm = SVM.load("E:\\TA\\New_Folder\\backgroud_substraction\\lib\\distF 40p\\res 40p\\BT_R_40p_0.xml");
//        svm = SVM.load("E:\\TA\\New_Folder\\backgroud_substraction\\lib\\distF 40p\\res 40p\\BT_R_40p_0.xml");
//        svm.load(cmbClassifier.getValue());
//        svm.load("‪E:\\TA\\file\\Res\\33333_0.xml");
        System.out.println(CvType.CV_32F);
        System.out.println(CvType.CV_32FC1);
    }

}
