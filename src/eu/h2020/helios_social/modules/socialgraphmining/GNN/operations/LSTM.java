package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

public class LSTM {
	private Matrix ui, wi, uf, wf, uo, wo, ug, wg;
	private Matrix tape_ui, tape_wi, tape_uf, tape_wf, tape_uo, tape_wo, tape_ug, tape_wg;
	
	public LSTM(int inputSize, int memorySize, int outputSize) {
		ui = new Matrix(memorySize, inputSize);
		uf = new Matrix(memorySize, inputSize);
		uo = new Matrix(memorySize, inputSize);
		ug = new Matrix(memorySize, inputSize);
		wi = new Matrix(memorySize, outputSize);
		wf = new Matrix(memorySize, outputSize);
		wo = new Matrix(memorySize, outputSize);
		wg = new Matrix(memorySize, outputSize);
		ui.setToRandom();
		uf.setToRandom();
		uo.setToRandom();
		ug.setToRandom();
		wi.setToRandom();
		wf.setToRandom();
		wo.setToRandom();
		wg.setToRandom();
	}
	
	public Tensor output(Tensor input, Tensor previousMemory, Tensor previousOutput) {
		Tensor i = Loss.sigmoid(ui.transform(input).selfAdd(wi.transform(previousOutput)));
		Tensor f = Loss.sigmoid(uf.transform(input).selfAdd(wf.transform(previousOutput)));
		Tensor o = Loss.sigmoid(uo.transform(input).selfAdd(wo.transform(previousOutput)));
		Tensor memoryGate = Loss.tanh(ug.transform(input).selfAdd(wg.transform(previousOutput)));
		Tensor memory = Loss.sigmoid(f.selfMultiply(previousMemory).selfAdd(i.selfMultiply(memoryGate)));
		return Loss.tanh(memory).selfMultiply(o);
	}
	
	public void startTape() {
		tape_ui = ui.zeroCopy();
		tape_uf = uf.zeroCopy();
		tape_uo = uo.zeroCopy();
		tape_ug = ug.zeroCopy();
		tape_wi = wi.zeroCopy();
		tape_wf = wf.zeroCopy();
		tape_wo = wo.zeroCopy();
		tape_wg = wg.zeroCopy();
	}
	
	public void updateTape(Tensor input, Tensor previousMemory, Tensor previousOutput, Tensor outputErrorGradient) {
		Tensor i = Loss.sigmoid(ui.transform(input).selfAdd(wi.transform(previousOutput)));
		Tensor f = Loss.sigmoid(uf.transform(input).selfAdd(wf.transform(previousOutput)));
		Tensor o = Loss.sigmoid(uo.transform(input).selfAdd(wo.transform(previousOutput)));
		Tensor memoryGate = Loss.tanh(ug.transform(input).selfAdd(wg.transform(previousOutput)));
		Tensor memory = Loss.sigmoid(f.multiply(previousMemory).selfAdd(i.multiply(memoryGate)));
		
		Tensor gradient_output = Loss.tanhDerivative(memory).multiply(outputErrorGradient);
		Tensor gradient_memory = Loss.sigmoidDerivative(f.multiply(previousMemory).selfAdd(i.multiply(memoryGate)));
		
		Tensor gradient_f = gradient_memory.multiply(previousMemory);
		Tensor gradient_i = gradient_memory.multiply(memoryGate);
		Tensor gradient_memoryGate = gradient_memory.multiply(i);
		
		//Matrix gradient 
		
		
		
	}
	
	public void endTape(double learningRate) {
		ui.selfAdd(tape_ui.selfMultiply(learningRate));
		uf.selfAdd(tape_uf.selfMultiply(learningRate));
		uo.selfAdd(tape_uo.selfMultiply(learningRate));
		ug.selfAdd(tape_ug.selfMultiply(learningRate));
		wi.selfAdd(tape_wi.selfMultiply(learningRate));
		wf.selfAdd(tape_wf.selfMultiply(learningRate));
		wo.selfAdd(tape_wo.selfMultiply(learningRate));
		wg.selfAdd(tape_wg.selfMultiply(learningRate));
		tape_ui = null;
		tape_uf = null;
		tape_uo = null;
		tape_ug = null;
		tape_wi = null;
		tape_wf = null;
		tape_wo = null;
		tape_wg = null;
	}
}
