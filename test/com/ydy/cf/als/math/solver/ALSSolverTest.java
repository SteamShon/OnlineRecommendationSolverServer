package com.ydy.cf.als.math.solver;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import com.google.common.collect.Maps;
import com.ydy.cf.common.VectorUtils;

public class ALSSolverTest {
  /**
   * 1) insert dummy matrix M
   * 2) insert user rating
   * 3) insert dummy matrix M'M
   * 4) run solver
   */
  private int numFeatures = 30;
  private int numItems = 100;
  private int topK = 10;
  private String testUserId = "1";
  private JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
  private Jedis jedis = pool.getResource();
  
  @Test
  public void test() {
    insertM();
    insertMtM();
    insertRatings();
    /*
    ALSSolverJob job = new ALSSolverJob();
    job.run(new String[]{
        testUserId, 
        String.valueOf(numFeatures),
        "0.065",
        "40",
        String.valueOf(topK)
    });
    */
    //jedis.flushAll();
  }
  
  private void insertM() {
    jedis.select(0);
    for (int r = 0; r < numItems; r++) {
      jedis.set(String.valueOf(r), VectorUtils.randomDenseVector(numFeatures).toString());
    }
  }
  private void insertMtM() {
    jedis.select(1);
    for (int r = 0; r < numFeatures; r++) {
      jedis.set(String.valueOf(r), VectorUtils.randomDenseVector(numFeatures).toString());
    }
  }
  
  private void insertRatings() {
    jedis.select(2);
    for (int i = 0; i < 10; i++) {
      Vector v = VectorUtils.randomSequentialVector(numFeatures);
      jedis.set(String.valueOf(i), v.toString());
    }
  }
  
}
