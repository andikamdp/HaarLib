/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package haarcoba;

import java.io.IOException;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

/**
 *
 * @author Andika Mulyawan
 */
public class HaarCoba extends Application {

    @Override
    public void start(Stage primaryStage) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FXMLLoader loader = new FXMLLoader();
        loader.setLocation(HaarCoba.class.getResource("haarCoba.fxml"));
        AnchorPane root = loader.load();
        Scene scane = new Scene(root);

        primaryStage.setScene(scane);
        primaryStage.show();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }

}
