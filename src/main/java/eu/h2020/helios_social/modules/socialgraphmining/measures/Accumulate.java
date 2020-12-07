package eu.h2020.helios_social.modules.socialgraphmining.measures;

import java.util.ArrayList;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This {@link Measure} is used to average the outcome of a base measure over given frames of time.
 * 
 * @author Emmanouil Krasanakis
 */
public class Accumulate implements Measure {
	private Measure baseMeasure;
	private ArrayList<Double> values = new ArrayList<Double>();
	private int memory;
	
	public Accumulate(Measure baseMeasure) {
		this(baseMeasure, 1000);
	}
	
	public Accumulate(Measure baseMeasure, int memory) {
		this.baseMeasure = baseMeasure;
		this.memory = memory;
	}
	
	private double getAverage() {
		if(values.isEmpty())
			return 0;
		double sum = 0;
		for(double val : values)
			sum += val;
		return sum / values.size();
	}

	@Override
	public double evaluateSend(SocialGraphMiner socialGraphMiner, Context context, Node dst) {
		double value = baseMeasure.evaluateSend(socialGraphMiner, context, dst);
		if(!Double.isFinite(value))
			return getAverage();
			//Utils.error("Value of base measure was not a finite value (was either NaN or Infity)");
		if(values.size()>=memory)
			values.remove(0);
		values.add(value);
		return getAverage();
	}

	@Override
	public double lastEvaluation() {
		return getAverage();
	}

}
