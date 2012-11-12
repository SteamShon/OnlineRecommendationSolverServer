package com.ydy.jetty.server;

import java.io.IOException;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.ydy.cf.solver.RecommendationSolver;

public class SolverHandler extends AbstractHandler {

  private static final Logger log = LoggerFactory.getLogger(SolverHandler.class);
  @Override
  public void handle(String target, Request baseRequest, HttpServletRequest request,
      HttpServletResponse response) throws IOException, ServletException {
    
    List<RecommendedItem> recs = Lists.newArrayList();
    RecommendationSolver solver = SolverHandlerFactory.createSolver(request);
    if (solver != null) {
      recs = solver.solveAll();
    }
    response.setContentType("text/html;charset=utf-8");
    
    response.setStatus(HttpServletResponse.SC_OK);
    baseRequest.setHandled(true);
    String output = buildOutput(recs);
    response.getWriter().append(output);
    log.info(output);
  }
  private String buildOutput(List<RecommendedItem> recs) {
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < recs.size(); i++) {
      if (i > 0) {
        sb.append(",");
      }
      sb.append(recs.get(i).getItemID() + ":" + recs.get(i).getValue());
    }
    return sb.toString();
  }
}
