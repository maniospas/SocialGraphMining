package eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

public class PeriodicReport implements Measure {
	private int interval = 1000;
	private int current = 0;
	private Measure measure;
	
	public PeriodicReport (Measure measure) {
		this.measure = measure;
	}
	
	public PeriodicReport (Measure measure, int interval) {
		this.measure = measure;
		this.interval = interval;
	}

	@Override
	public double evaluateSend(SocialGraphMiner socialGraphMiner, Context context, Node dst) {
		double value = measure.evaluateSend(socialGraphMiner, context, dst);
		current += 1;
		if(current % interval==0) {
			measure.lastEvaluation();
			System.out.println("#"+current+": "+value);
		}
		return value;
	}
	
	@Override
	public double lastEvaluation() {
		return measure.lastEvaluation();
	}
}
