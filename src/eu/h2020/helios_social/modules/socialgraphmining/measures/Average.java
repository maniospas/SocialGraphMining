package eu.h2020.helios_social.modules.socialgraphmining.measures;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This {@link Measure} is used to average the outcome of a base measure over all frames of time
 * (is a computationally tractable version of {@link Accumulate} for infinite memory.
 * 
 * @author Emmanouil Krasanakis
 */
public class Average implements Measure {
	private Measure baseMeasure;
	private double sum;
	private int samples;
	
	public Average(Measure baseMeasure) {
		this.baseMeasure = baseMeasure;
	}
	
	private double getAverage() {
		if(samples==0)
			return 0;
		return sum / samples;
	}

	@Override
	public double evaluateSend(SocialGraphMiner socialGraphMiner, Context context, Node dst) {
		double value = baseMeasure.evaluateSend(socialGraphMiner, context, dst);
		if(!Double.isFinite(value))
			return getAverage();
		sum += value;
		samples += 1;
		return getAverage();
	}

}
