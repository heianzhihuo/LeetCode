package asyn;


public class Demo1 extends BaseDemo {
	
	private final Object lock = new Object();
	private int semaphore = 0;
	
	@Override
	public void callback(long response) {
		// TODO Auto-generated method stub
		System.out.println("得到结果");
		System.out.println(response);
		System.out.println("调用结束");
		synchronized (lock) {
			semaphore++;
			lock.notifyAll();
		}
	}

	public static void main(String[] args) {
		Demo1 demo1 = new Demo1();
		demo1.call();
		//这里加锁存在一个问题,如果子线程比主线程先执行完成
		//即callback先执行完成，此时已经调用了lock.notifyAll()
		//而主线程后执行lock.wait()，此时将发生无限期等待
		//因此改进后，添加了一个条件，这种无限期等待
		try {
			Thread.sleep(5*1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		synchronized (demo1.lock) {
			try {
				while(demo1.semaphore==0)
					demo1.lock.wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("主线程内容");
	}

}
