package asyn;

public abstract class BaseDemo {
	protected AsyncCall asyncCall = new AsyncCall();
	public abstract void callback(long response);
	public void call() {
		System.out.println("发起调用");
		asyncCall.call(this);//在call这里生了一个新的线程，即使call的线程还未结束，将继续执行后面的调用返回
		System.out.println("调用返回");
	}
}
