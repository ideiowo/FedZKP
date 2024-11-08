pragma circom 2.0.0;

template Sum(N, M) {
    signal input values[N * M];
    signal output sums[M];

    // 使用臨時變數進行累加
    var temp;

    // 對每個位置上的值進行累加
    for (var i = 0; i < M; i++) {
        temp = 0;
        for (var j = 0; j < N; j++) {
            temp += values[i + j * M];
        }
        sums[i] <== temp;
    }
}

component main = Sum(10, 6399); // N = 客戶數量，M = 參數數量
