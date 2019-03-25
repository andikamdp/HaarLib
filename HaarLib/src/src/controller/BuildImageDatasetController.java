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
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.Mat;
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
    private Button btnStartCamera;
    @FXML
    private TextField txtSaveFileLocation;
    @FXML
    private Button btnCreateFolder;
    @FXML
    private AnchorPane apGetImage;
    @FXML
    private TextField txtBoxClassName;
    @FXML
    private TextField txtBoxImageCount;
    @FXML
    private TextField txtBoxClassCount;
    @FXML
    private ImageView MainFrame;
    @FXML
    private Button btnBrowseSaveFile;
//
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private int i, j;
    private MainAppController mainAppController;
    private File imgDir, imgLblDir;
    private Alert alert;
    private ImageView layarEdge;
    private ImageView layarBW;

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
        txtBoxImageCount.setText(String.valueOf(i));
        imgDir = null;
        imgLblDir = null;
        alert = new Alert(Alert.AlertType.ERROR);

    }

    /**
     * ######################################################################
     * Method awal untuk membuka kamera dan memanggil method
     * var:
     * boolean cameraActive :
     * VideoCapture capture :
     * Runnable frameGrabber :
     * Mat frame :
     * ScheduledExecutorService timer :
     * Button btnStartCamera :
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        i = 0;
        j = Integer.valueOf(txtBoxImageCount.getText());
        if (!txtBoxClassName.getText().equals("") && !txtSaveFileLocation.getText().equals("") && imgDir.exists() && imgLblDir.exists()) {
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
        } else {
            alert.show();
        }
    }

    /**
     * ######################################################################
     * OnClick Action Untuk mencari Folder untuk menyimpan gambar
     * var:
     *
     */
    @FXML
    private void browsSaveFileOnClick(ActionEvent event
    ) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setInitialDirectory(imgDir);
        brows.setTitle("Buka Folder Data Training");
        imgDir = brows.showDialog(apGetImage.getScene().getWindow());
        if (imgDir != null) {
            txtSaveFileLocation.setText(imgDir.getAbsolutePath());
        }
    }

    /**
     * ######################################################################
     * OnCllick Action creating new folder
     * var:
     * File file : lable file location
     */
    @FXML
    private void createFolderOnClick(ActionEvent event
    ) {
        File file = new File(imgDir + "\\" + txtBoxClassName.getText());

        if (!file.exists() && !txtBoxClassName.getText().equals("")) {
            file.mkdir();
        } else {
            alert = new Alert(Alert.AlertType.ERROR);
            alert.setContentText("Directory Already Exist");
            alert.show();
        }
        imgDir = new File(imgDir.getAbsolutePath());
        txtBoxClassCount.setText(String.valueOf(imgDir.list().length));
        imgLblDir = file;
    }

    @FXML
    private void getFramePointOnClick(MouseEvent event
    ) {
    }

    /**
     * ######################################################################
     *
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
     * Method untuk memperoleh Frame dari kamera
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
     * method untuk menyimpan gambar dalam kotak merah MainFrame
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
                    Imgcodecs.imwrite(imgLblDir.getAbsolutePath() + "\\" + txtBoxClassName.getText() + "_" + i + ".jpg", hand);
                    i++;
                    txtBoxImageCount.setText(String.valueOf(j - i));
                }
            } else {
                alert.setContentText("Directory Not Exist");
                alert.show();
            }
        } catch (Exception e) {
            System.out.println("start(Mat frame) " + e);
        }

    }

    void setMainController(MainAppController aThis) {
        this.mainAppController = aThis;
    }

}
