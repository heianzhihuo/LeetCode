
public abstract class Semaphore {
	private int value;
	public Semaphore() {
	}
	public Semaphore(int intial) {
		if(intial>=0)
			value = intial;
		else
			throw new IllegalArgumentException("intial<0");
	}
	public final synchronized void P() throws InterruptedException{
		while (value==0)
			wait();
		value--;
			
	}
	
	public final synchronized void Vc() {
		value++;
		notifyAll();
	}
	
	public final synchronized void Vb() {
		value++;
		notifyAll();
		if(value>1)
			value = 1;
	}
	
	public abstract void V();
	
	@Override
	public String toString() {
		return ".value="+value;
	}
}
