import java.util.Scanner;

public class LeetCode4 {
	
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
            int d = in.nextInt();
            String a = in.nextLine();
            String b = in.nextLine();
            String res = "";
            String aa[] = a.split(",");
            String bb[] = b.split(",");
            int n1 = aa.length;
            int n2 = bb.length;
            int i,j,k;
            i = 0;
            while(i+d<n1 && i+d<n2){
                for(k=0;k<d;k++)
                    res += aa[i+k] + ",";
                for(k=0;k<d;k++)
                    res += bb[i+k] + ",";
                i += d;
            }
            if(i+d<n1){
                for(k=0;k<d;k++)
                    res += aa[i+k] + ",";
                for(k=0;k<n2-d;k++)
                    res += bb[i+k] + ",";
                i+=d;
                for(;i<n1;i++)
                    res += aa[i+k] + ",";
            }else{
                for(k=0;k<n1-d;k++)
                    res += aa[i+k] + ",";
                for(;i<n2;i++)
                    res += bb[i+k] + ",";
            }
            System.out.println(res.substring(res.length()-1));
        }
    }
	
	/*public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
		while (in.hasNextInt()) {// 注意while处理多个case
			int q = in.nextInt();
			for (int i = 0; i < q; i++) {
				int m = in.nextInt();
				int n = in.nextInt();
				if (m % 2 == 0 && n % 2 == 0) {
					System.out.println(-(n - m) / 2 + m);
				} else if (m % 2 == 1 && n % 2 == 1) {
					System.out.println((n - m) % 2 + m);
				} else if (m % 2 == 0 && n % 2 == 1) {
					System.out.println(-(n - m + 1) % 2);
				} else {
					System.out.println((n - m + 1) % 2);
				}
			}
		}

	}*/
}
