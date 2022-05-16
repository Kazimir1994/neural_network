package kazzimir.bortnik;

import java.io.FileWriter;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        FileWriter myWriter = new FileWriter("filename.txt");
        myWriter.write("Files in Java might be tricky, but it is fun enough!");
        myWriter.write("\r\n");
        myWriter.write("Files in Java might be tricky, but it is fun enough2");
        myWriter.close();
    }
}
