/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import src.utils.Utils;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class BuildImageDatasetController implements Initializable {

    @FXML
    private Button btnBrowseSaveLocation;
    @FXML
    private TextField txtBoxImageClassName;
    @FXML
    private TextField txtBoxNumberOfImage;
    @FXML
    private Label lblImageSaveLocation;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnCreateFolder;
    @FXML
    private BorderPane apGetImage;
    @FXML
    private TextField txtBoxClassCount;
    @FXML
    private ImageView MainFrame;
//
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private int i, j;
    private MainAppController mainAppController;
    private File imgDir, imgLblDir;

    /**
     * Initializes the controller class.
     *
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        i = 0;
        txtBoxNumberOfImage.setText(String.valueOf(i));
        imgDir = null;
        imgLblDir = null;
        Mat image = new Mat(480, 640, CvType.CV_8UC3, new Scalar(255, 255, 255));
        image = Preprocessing.drawRect(image);
        updateImageView(MainFrame, image);
    }

    /**
     * ######################################################################
     * Method awal untuk memulai membuka kamera dan memulai proses pengambilan gambar.
     * Method ini pun menjalankan thread untuk terus melakukan proses pengambilan gambar.
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        i = 0;
        j = Integer.valueOf(txtBoxNumberOfImage.getText());
        if (!txtBoxImageClassName.getText().equals("") && !lblImageSaveLocation.getText().equals("") && imgDir.exists() && imgLblDir.exists()) {
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

                    timer.scheduleAtFixedRate(frameGrabber,
                            0, 33,
                            TimeUnit.MILLISECONDS);
                    btnStartCamera.setText(
                            "stop Camera");
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
    }

    /**
     * ######################################################################
     * Method untuk mencari lokasi direktori untuk menyimpan data gambar.
     */
    @FXML
    private void browseSaveLocationOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setInitialDirectory(imgDir);
        brows.setTitle("Open Folder");
        imgDir = brows.showDialog(apGetImage.getScene().getWindow());
        if (imgDir != null) {
            lblImageSaveLocation.setText(imgDir.getAbsolutePath());
        }
    }

    /**
     * ######################################################################
     * Method untuk membuat folder baru dari lokasi penyimpanan yang dipilih dengan nama sesuai txtBoxClassName.
     */
    @FXML
    private void createFolderOnClick(ActionEvent event
    ) {
        File file = new File(imgDir + "\\" + txtBoxImageClassName.getText());

        if (!file.exists() && !txtBoxImageClassName.getText().equals("")) {
            file.mkdir();
        }
        imgDir = new File(imgDir.getAbsolutePath());
        txtBoxClassCount.setText(String.valueOf(imgDir.list().length));
        imgLblDir = file;
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
     * update tampilan pada frame
     * var: Method untuk mengganti gambar dari main frame.
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
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

    /**
     * ######################################################################
     * Method untuk memperoleh gambar dari kamera.
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
     * Method untuk menambahkan ROI pada gambar dan memulai menyimpan gambar pada main frame.
     *
     */
    private void start(Mat frame) {
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        updateImageView(MainFrame, frame);
        Mat hand = Preprocessing.getBox(frame.clone());
//        updateImageView(layarEdge, hand);
        //
        try {
            if (imgDir.exists() && imgLblDir.exists()) {
                if (i < j) {
                    Imgcodecs.imwrite(imgLblDir.getAbsolutePath() + "\\" + txtBoxImageClassName.getText() + "_" + i + ".jpg", hand);
                    i++;
                    txtBoxNumberOfImage.setText(String.valueOf(j - i));
                }
            }
        } catch (Exception e) {
            System.out.println("start(Mat frame) " + e);
        }

    }

    /**
     * ######################################################################
     * Method untuk menentukan Main Controller dari tampilan.
     *
     */
    void setMainController(MainAppController aThis) {
        this.mainAppController = aThis;
    }

}
