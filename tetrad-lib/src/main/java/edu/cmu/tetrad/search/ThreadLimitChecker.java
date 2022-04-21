package edu.cmu.tetrad.search;

/*
 * Java Program to check when JVM will throw
 * "java.lang.OutOfMemoryError: unable to create new native thread" error.
 */

public class ThreadLimitChecker{

  public static void main(String[] args) {
    int count = 0;
    while (true) {
      count++;     
      new Thread(new Runnable() {
        public void run() {
          try {
            Thread.sleep(10000000);
          } catch (InterruptedException e) {
          }
        }
      }).start();
      System.out.println("Thread #:" + count);
    }
  }
}