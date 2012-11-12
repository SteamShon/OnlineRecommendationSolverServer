package com.ydy.jetty.server;

import org.eclipse.jetty.server.Server;

import com.ydy.cf.model.MatrixLoader;



public class SolverServer {
  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      throw new Exception("Invalid number of arguments: [datafile] [solver type] [port]");
    }
    String dataFile = args[0];
    int solverType = Integer.parseInt(args[1]);
    int port = Integer.parseInt(args[2]);
    
    MatrixLoader.dataFile = dataFile;
    MatrixLoader.solverType = solverType;

    MatrixLoader.getInstance();

    Server server = new Server(port);
    
    server.setHandler(new SolverHandler());
   
    server.start();
    server.join();
}
}
