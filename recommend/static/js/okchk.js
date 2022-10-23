function okchk(){
    var arr = [];

    $("input[id=flexCheckDefault]:checked").each(function(i){
        arr.push($(this).val());
    });

    console.log(arr);

    alert("배열 생성 완료");
}