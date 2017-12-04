
/**
 * Created by Piotr Smuga on 28.11.2017.
 */

public class Main {

    public static void main(String[] args) {
        HebbProgram hebb = new HebbProgram(0.1,0.1);
        hebb.Train();
        hebb.Test();
    }
}