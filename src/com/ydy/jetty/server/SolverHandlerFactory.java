package com.ydy.jetty.server;

import javax.servlet.http.HttpServletRequest;

import org.apache.mahout.math.Vector;

import com.ydy.cf.common.VectorUtils;
import com.ydy.cf.model.MatrixLoader;
import com.ydy.cf.solver.RecommendationSolver;
import com.ydy.cf.solver.impl.AlternatingLeastSquaresImplicitSolver;
import com.ydy.cf.solver.impl.AlternatingLeastSquaresSolver;
import com.ydy.cf.solver.impl.ItemBasedRecommendationSolver;
import com.ydy.cf.solver.impl.RandomWalkSolver;
/**
 * Handler Factory.
 * 
 * @author ydy
 *
 */
public class SolverHandlerFactory {
  // default parameters for ALS-WR
  private static final double lambda = 0.065;
  private static final int alpha = 40;
  
  public static RecommendationSolver createSolver(HttpServletRequest request) {
    RecommendationSolver solver = null;
    // parse request input
    Vector ratingsVector = null;
    String userId = request.getParameter("user_id").toString().trim();
    String ratings = request.getParameter("ratings").toString().trim();
    int solverType = Integer.parseInt(request.getParameter("solver_type").toString().trim());
    int iteration = request.getParameter("iteration") == null ? 10 :
      Integer.parseInt(request.getParameter("iteration").toString());
    int topK = Integer.parseInt(request.getParameter("topK").toString().trim());
    MatrixLoader loader = MatrixLoader.getInstance();
    
    ratingsVector = VectorUtils.parseVector(ratings, loader.getRatingDimension());
    
    switch (solverType) {
    case 0:
      solver = new AlternatingLeastSquaresImplicitSolver(userId, ratingsVector, lambda, alpha, loader, topK);
      break;
    case 1:
      solver = new AlternatingLeastSquaresSolver(userId, ratingsVector, lambda, loader, topK);
      break;
    case 2:
      solver = new ItemBasedRecommendationSolver(userId, ratingsVector, loader, topK);
      break;
    case 3:
      RandomWalkSolver.ITERATION = iteration;
      solver = new RandomWalkSolver(userId, ratingsVector, loader, topK);
      break;
    }
    return solver;
  }
}
