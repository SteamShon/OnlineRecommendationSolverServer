package com.ydy.cf.solver.impl;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.ydy.cf.common.VectorUtils;
import com.ydy.cf.model.MatrixLoader;

public class RandomWalkSolver extends AbstractRecommendationSolver {
  private static final Logger log = LoggerFactory.getLogger(RandomWalkSolver.class);
  private DataModel dataModel;
  private FastByIDMap<Float> userWeightSums;
  private FastByIDMap<Float> itemWeightSums;
  
  private static float MIN_PROB = (float) 1.0e-10;
  public static int ITERATION =10;
  private static int threadPoolSize = 3;
  private static boolean USE_MAX_PRODUCT = false;
  private final float MAX_RATE = 10f;
  
  public RandomWalkSolver(String userId, Vector userRatings, MatrixLoader loader, int numRecommendations) {
    super(userId, userRatings, loader, numRecommendations);
    dataModel = loader.getDataModel();
    userWeightSums = loader.getUserWeightSums();
    itemWeightSums = loader.getItemWeightSums();
    log.info(RandomWalkSolver.class.getName() + "\t" + userId + "\t" + userRatings);
  }
  
  @Override
  public List<RecommendedItem> solveAll() {
    List<RecommendedItem> recs = Lists.newArrayList();
    // solver ignore tasteException and Interrupted exception
    try {
      Map<Integer, Float> PI = walk();
      Map<Integer, Boolean> alreadyExist = Maps.newHashMap();
      Iterator<Vector.Element> iter = userRatings.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element e = iter.next();
        alreadyExist.put(e.index(), true);
      }
      TopK<RecommendedItem> topK = new TopK<RecommendedItem>(numRecommendations, VectorUtils.BY_PREFERENCE_VALUE);
      for (Entry<Integer, Float> pi : PI.entrySet()) {
        if (alreadyExist.containsKey(pi.getKey())) {
          continue;
        }
        topK.offer(
            new GenericRecommendedItem(pi.getKey(), pi.getValue()));
        
      }
      recs = topK.retrieve();
    } catch (TasteException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return recs;
  }
  
  /*
   * set initial seed probability for items this user rated.
   * this can be overrided for different purpose.
   */
  private Map<Integer, Float> initProbs(Vector ratings) {
    Map<Integer, Float> PI = Maps.newHashMap();
    Iterator<Vector.Element> iter = ratings.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element e = iter.next();
      //PI.put(e.index(), (float)e.get() / MAX_RATE);
      PI.put(e.index(), 1f);
    }
    return PI;
  }
  
  /*
   * spread formulation in simrank++. never used for this time.
   */
  @SuppressWarnings("unused")
  private double getSpreads(Vector seedItemIds, int targetItemId) throws TasteException {
    double ret = 0.0;
    int common = 0;
    Iterator<Vector.Element> seeds = seedItemIds.iterateNonZero();
    while (seeds.hasNext()) {
      Vector.Element seed = seeds.next();
      //common += dataModel.getNumUsersWithPreferenceFor(seed.index(), targetItemId);
    }
    for (int i = 0; i <= common; i++) {
      ret += Math.pow(2.0, -1.0 * i);
    }
    return ret;
  }
  /* 
   * execute actual propagations. note that each user_id, item_id 
   * can be run simultaneously without lock since they are mutual exclusive and read only.
   */
  private static class WalkerThread extends Thread {
    public static Map<Integer, Float> userProbs;
    public static Map<Integer, Float> itemProbs;
    public static DataModel model;
    public static FastByIDMap<Float> userSums;
    public static FastByIDMap<Float> itemSums;
    
    private final List<Integer> ids;
    private boolean forUser;
    
    WalkerThread(List<Integer> ids, boolean forUser) {
      this.ids = ids;
      this.forUser = forUser;
    }
   
    @Override
    public void run() {
      for (Integer id : ids) {
        float maxProb = 0f;
        float prob = 0f;
        PreferenceArray others;
        try {
          // walk on this threads users or items
          others = forUser ? model.getPreferencesFromUser(id) : model.getPreferencesForItem(id);
          // to normalize edge sums
          float weightSum = forUser ? userSums.get(id) : itemSums.get(id);
          
          for (Preference other : others) {
            int otherId = (int) (forUser ? other.getItemID() : other.getUserID());
            float weight = other.getValue() / weightSum;
            
            Map<Integer, Float> priors = forUser ? itemProbs : userProbs;
            if (priors.containsKey(otherId)) {
              float cur = priors.get(otherId) * weight;
              prob += cur;
              maxProb = Math.max(maxProb, cur);
            }
          }
          Map<Integer, Float> posteriors = forUser ? userProbs : itemProbs;
          if (USE_MAX_PRODUCT) {
            // max-product algorithm.
            if (maxProb > MIN_PROB) {
              posteriors.put(id, maxProb);
            }
          } else {
            // label propagation.
            // absorb state
            if (prob > MIN_PROB) {
              if (posteriors.containsKey(id)) {
                posteriors.put(id, Math.max(posteriors.get(id), prob));
              } else {
                posteriors.put(id, prob);
              }
            }
          }
        } catch (TasteException e) {
          e.printStackTrace();
        }
      }
    }
  }
  /*
   * nesty hashing function for load balance between threads(id modular N)
   */
  private Map<Integer, List<Integer>> splitIdsForThreads(LongPrimitiveIterator ids, int numThreads) {
    Map<Integer, List<Integer>> pools = Maps.newHashMap();
    while (ids.hasNext()) {
      long id = ids.next();
      int mod = (int)id % numThreads;
      if (pools.containsKey(mod) == false) {
        pools.put(mod, new ArrayList<Integer>());
      }
      pools.get(mod).add((int)id);
    }
    return pools;
  }
  /*
   * runner for thread pools.
   */
  private Map<Integer, Float> walk() throws TasteException, InterruptedException {
    Map<Integer, Float> PI = initProbs(this.userRatings);
    Map<Integer, Float> PU = Maps.newHashMap();
    WalkerThread.model = dataModel;
    WalkerThread.userSums = userWeightSums;
    WalkerThread.itemSums = itemWeightSums;
    WalkerThread.userProbs = PU;
    WalkerThread.itemProbs = PI;
    
    Map<Integer, List<Integer>> userIdPools = splitIdsForThreads(dataModel.getUserIDs(), threadPoolSize);
    Map<Integer, List<Integer>> itemIdPools = splitIdsForThreads(dataModel.getItemIDs(), threadPoolSize);
    for (int iter = 0; iter < ITERATION; iter++) {
      log.info("Users iteration: " + iter);
      // calculate posterior probability for users
      for (Entry<Integer, List<Integer>> e : userIdPools.entrySet()) {
        WalkerThread thread = new WalkerThread(e.getValue(), true);
        thread.start();
        thread.join();
      }
      log.info("Items iteration: " + iter);
      // calculate posterior probability for items
      for (Entry<Integer, List<Integer>> e : itemIdPools.entrySet()) {
        WalkerThread thread = new WalkerThread(e.getValue(), false);
        thread.start();
        thread.join();
      }
    }   
    return PI;
  }
  
}
