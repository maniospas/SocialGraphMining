package eu.h2020.helios_social.modules.socialgraphmining.measures;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * @author Emmanouil Krasanakis
 */
public class DiscoveryExactRank implements Measure {
	private int requiredCandidates;
	
	public DiscoveryExactRank(int requiredCandidates) {
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
			if(value > dstValue)
				countLargerValues += 1;
		return countLargerValues==requiredCandidates?1:0;
	}
}
