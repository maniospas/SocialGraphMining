package eu.h2020.helios_social.modules.socialgraphmining.measures;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This {@link Measure} provides a HitRate@k evaluation, which measures whether the occurred interactions lie among the top k recommendations
 * provided by {@link SocialGraphMiner#recommendInteractions(Context)}. Ties are resolved towards not favoring the
 * recommendation, so that for example all-zero recommendations are evaluated as failed ones. Since this {@link Measure}
 * is used to assess only one prediction instance at a time, its assessment is either 0 or 1 and needs be aggregated
 * across multiple interactions using another measure, such as {@link Accumulate}. Since it makes no sense to assess recommendations
 * between too few nodes (e.g. less than k), the {@link #evaluateSend(SocialGraphMiner, Context, Node)} method returns a NaN value
 * if there are fewer than a preselected
 * number of predictions.
 * 
 * @author Emmanouil Krasanakis
 */
public class HitRate implements Measure {
	private int k;
	private int requiredCandidates;
	
	/**
	 * Default constructor for calculating HitRate@3.
	 * @see #HitRate(int)
	 */
	public HitRate() {
		this(3, 6);
	}

	/**
	 * Constructor that creates for calculating HitRate@k when there are at least 2k or more recommendations.
	 * @param k The number of top recommendations in which to check that occurring interactions reside.
	 * @see #HitRate(int, int)
	 */
	public HitRate(int k) {
		this(k, 2*k);
	}
	
	/**
	 * Constructor that creates a HitRate@k evaluation.
	 * @param k The number of top recommendations in which to check that occurring interactions reside.
	 * @param requiredCandidates The number of recommendations that need be available for the evaluation to be valid.
	 */
	public HitRate(int k, int requiredCandidates) {
		this.k = k;
		this.requiredCandidates = requiredCandidates;
	}

	@Override
	public double evaluateSend(SocialGraphMiner socialGraphMiner, Context context, Node dst) {
		HashMap<Node, Double> maps = socialGraphMiner.recommendInteractions(context);
		if(!maps.containsKey(dst) || maps.size() < requiredCandidates)
			return Double.NaN;
		double dstValue = maps.get(dst);
		if(dstValue==0)
			return Double.NaN;
		int countLargerValues = 0;
		for(double value : maps.values())
			if(value >= dstValue)
				countLargerValues += 1;
		return countLargerValues<=k?1.:0.;
	}

}
