package eu.h2020.helios_social.modules.socialgraphmining.experiments;

import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.storage.NoStorage;
import eu.h2020.helios_social.modules.socialgraphmining.diffusion.PPRMiner;
import eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation.Device;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.datasets.Dataset;
import mklab.JGNN.datasets.Datasets;
import mklab.JGNN.models.IdConverter;

public class DiffusionSimulation {
	private static double fractionOfKnownLabels = 0.1;
	
	protected static PPRMiner createMiner(String deviceName, int classificationId) {
		ContextualEgoNetwork cen = ContextualEgoNetwork.createOrLoad(new NoStorage("NOFILESYSTEM\\"), deviceName, null);
		return new PPRMiner("prediction diffusion", cen, 
				Math.random()<fractionOfKnownLabels?new DenseTensor(15).put(classificationId, 1):new DenseTensor(15));
	}
	
	public static long argmax(Tensor tensor) {
		double maxValue = 0;
		long maxPos = tensor.size();
		for(long pos : tensor) {
			double value = tensor.get(pos);
			if(value > maxValue) {
				maxValue = value;
				maxPos = pos;
			}
		}
		return maxPos;
	}

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CoraGraph();
		HashMap<String, Device> devices = new HashMap<String, Device>();
		IdConverter vectorization = new IdConverter();
		for(int epoch=0;epoch<50;epoch++) {
			for(Entry<String, String> interaction : dataset.getInteractions()) {
				String u = interaction.getKey();
				String v = interaction.getValue();
				if(u.equals(v))
					continue;
				if(!devices.containsKey(u))
					devices.put(u, new Device(createMiner(u, vectorization.getOrCreateId(dataset.getLabel(u)))));
				if(!devices.containsKey(v))
					devices.put(v, new Device(createMiner(v, vectorization.getOrCreateId(dataset.getLabel(v)))));
				devices.get(u).send(devices.get(v));
				//System.out.println(((PPRMiner)devices.get(u).getMiner()).getNormalizedSmoothedPersonalization());
			}
			int acc = 0;
			for(String u : devices.keySet()) 
				if(argmax(((PPRMiner)devices.get(u).getMiner())
						.getSmoothedPersonalization(devices.get(u).getMiner().getContextualEgoNetwork().getCurrentContext()))
						== vectorization.getOrCreateId(dataset.getLabel(u)))
					acc += 1;
			System.out.println("Epoch "+epoch+" accuracy "+acc/(float)devices.size());
		}
	}
}
