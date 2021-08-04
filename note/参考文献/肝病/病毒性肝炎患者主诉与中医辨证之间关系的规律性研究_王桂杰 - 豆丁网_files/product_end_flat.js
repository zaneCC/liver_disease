//下载文档
function downLoadNew(productId,fn){
	var random = new Date().getTime();
	jQuery.ajax({
			type:'GET',
			url:"/newEnd/down_load_new_v1.do",
			dataType:"html",
            async: false,
			context: document.body,
			data: "pid="+productId+"&fn="+fn+"&flag=down&datetime="+random,
			success:function(data){
				if(data){
					var index = data.indexOf("1|");
					if(index == 0){//文档可直接下载
						
						if(isFreeProduct == "1" && isZuZhi == "1"){//免费文档下载加限制
							mailDialogAlert(login_email,landingEmail,domailUrl);
							return false;
						}

						var downPath = data.split("|")[1];
						var Sys = {};
						var ua = navigator.userAgent.toLowerCase();
						var s;
							(s = ua.match(/version\/([\d.]+).*safari/)) ? Sys.safari = s[1] : 0;

						if (typeof history.pushState == "function") {
							if (Sys.safari){
								downloadSuccess(productId);
								location.href=downPath;
								return false;
							}
							var name=downPath.substring(downPath.indexOf("filename=")+9,downPath.length);
							var name=decodeURI(decodeURI(name))
							download(downPath, function(blob){saveAs(blob,name);},productId);
							return false;
						}else{
							downloadSuccess(productId);
							location.href=downPath;
							return false;
						}
					}
					
					var oHtml = data;
					jQuery(this).append(oHtml);
					//充值  购买  企业vip
					if(idflag == "downLoadTips1" || idflag == "downLoadTips2" || idflag == "downLoadTips4"){
						dialogBoxShadow();
					}
					if(idflag != 'downLoadTips3'&& idflag != 'downLoadTips7'){
						setObjCenter(idflag);
					}
					else{
						if(docinReader.config.isBuy == 0){
							//有一种文档是免费下载，付费阅读的，这种下载后不会加载剩下的
							if(specialProd == "true"){ 
							
								//下载成功后可阅读全文
								if(jQuery("#j_pay").length>0){
									jQuery("#j_pay").removeClass('orange').addClass('gray').removeAttr('onclick');
									jQuery("#j_pay").html('<span class="ico_mini ico_buy"></span>已付费'); 
								}								
								//付费后加载剩下的
								docinReader.config.isBuy = 2;
								docinReader.createDiv();
								docinReader.setReaderLogin();
								if(jQuery(".preview_end").length>0){
									jQuery(".preview_end").remove();
								}
								
							}
						}
					}
					if(idflag == "downLoadTips1"){
						//下载、购买提示
						if(jQuery(".zk").length>0){
							jQuery(".zk").click(function(){
								var $index=jQuery(".zk").index(this);
								jQuery(".cz_btn").hide();
								jQuery(".cz_btn").eq($index).show();
								jQuery(".zk").show();
								jQuery(this).hide();
							});
						}
					}
				}
			}
		});
}
//VIP免费下载
function downLoadNewVip(productId,fn){
	var datetime =  new Date().getTime();
	closeDownLoadDiv('4');
	jQuery.ajax({
			type:'GET',
			url:"/newEnd/vipFreeDownload_new.do",
			context: document.body,
			data: "pid="+productId+"&fn="+fn+"&datetime="+datetime,
			success:function(data){
				if(data){
					var oHtml = data;
					jQuery(this).append(oHtml);
					//dialogBoxShadow();
					//setObjCenter(idflag);
				}
			}
		});
}
//点击下载，弹出下载成功  /newEnd/down_load_success.do?pid=xxx&clickType=2

function downloadSuccess(productId){
	jQuery.ajax({
		type:'GET',
		url:"/newEnd/down_load_success.do",
		async: false,
		context: document.body,
		data: "pid="+productId+"&clickType=2",
		success:function(data){
			if(data){
				var oHtml = data;
				jQuery(this).append(oHtml);
				dialogBoxShadow();
				//动态开始加载收藏该文档的用户也收藏
				sortCourseTime_ASC(2);
				setObjCenter("addFavSuccess");
				jQuery("#addFavSuccess").show();
				docinCopyData('copy1','docUrl2');
			}
		}
	});
}
//收藏到书房
function addDocToBookshop(pid){
	
	if(isZuZhi == "1"){
		mailDialogAlert(login_email,landingEmail,domailUrl)
		return;
	}
	
	
	if(jQuery("#selectFolder").length>0){
		if(jQuery("#dialogBoxShadow").length == 0){
			dialogBoxShadow();
		}
		jQuery("#selectFolder").show();
		return false;
	}
	//按钮变为已收藏
	jQuery("#addBookShop").removeClass('green').addClass('gray').html('<span class="ico_mini ico_addfav"></span>已收藏');
	jQuery.ajax({
		type:'GET',
		url:"/app/dwrutil/showFavoriteCode.do?from=newend&pid="+pid+"&t=" + new Date().getTime(),
		success:function(data){
			if(data != null && data.indexOf('div') > -1){
				if(jQuery("#userfavoriteproductdiv") != null){
					jQuery("#userfavoriteproductdiv").remove();
				}
				var oHtml = data;
				jQuery(document.body).append(oHtml);
			}else{
				jQuery.ajax({
					type:'GET',
					url:"/newEnd/getUserFolders_New.do",
					context: document.body,
					data: "productId="+pid+"&t=" + new Date().getTime(),
					success:function(data){
						if(data){
							var oHtml = data;
							jQuery(this).append(oHtml);
							dialogBoxShadow();
							setObjCenter("selectFolder");

						}
					}
				});
			}
		}
	});
}
//购买付费
function purchaseProduct(pid,uid,flag){
	jQuery.ajax({
			type:'GET',
			url:"/newEnd/purchaseProduct_new.do",
			context: document.body,
			data: "pid="+pid+"&purchaserId="+uid+"&flag=buy",
			success:function(data){
				if(docinReader.config.isBuy == 1){return false;}
				if(data){
					var oHtml = data;
					jQuery(this).append(oHtml);
					if(isNewTradeProd != null && isNewTradeProd == "true"){
						window.location.reload();
						return;
					}
					closeDownLoadDiv('2');
					//dialogBoxShadow();
					if(idflag != 'downLoadTips7'){
						setObjCenter(idflag);
					}
					if(idflag == "downLoadTips6"){
						//docinCopyData('share_copy','shareDocUrl');
					}
					if(jQuery("#j_pay").length>0){
						jQuery("#j_pay").removeClass('orange').addClass('gray').removeAttr('onclick');
						jQuery("#j_pay").html('<span class="ico_mini ico_buy"></span>已付费'); 
					}
					//付费后加载剩下的
					docinReader.config.isBuy = 2;
					docinReader.createDiv();
					docinReader.setReaderLogin();
					if(jQuery(".preview_end").length>0){
						jQuery(".preview_end").remove();
					}
				}
			}
		});
}
//购买下载
function purchase(pid,uid,flag){
	
	jQuery.ajax({
		type:'POST',
		url:"/newEnd/purchaseProduct_new.do",
		dataType:"html",
        async: false,
		context: document.body,
		data: "pid="+pid+"&purchaserId="+uid+"&flag="+flag,
		success:function(data){
		closeDownLoadDiv('2');
			if(data){
				var index = data.indexOf("1|");
				if(index == 0){//文档可直接下载
					var downPath = data.split("|")[1];
					var Sys = {};
					var ua = navigator.userAgent.toLowerCase();
					var s;
						(s = ua.match(/version\/([\d.]+).*safari/)) ? Sys.safari = s[1] : 0;

					if (typeof history.pushState == "function") {
						if (Sys.safari){
							downloadSuccess(productId);
							location.href=downPath;
							return false;
						}
						var name=downPath.substring(downPath.indexOf("filename=")+9,downPath.length);
						var name=decodeURI(decodeURI(name))
						download(downPath, function(blob){saveAs(blob,name);},productId);
						return;
					}else{
						downloadSuccess(pid);
						location.href=downPath;
						return;
					}
				}
			}
			downLoadNew(pid,'');
		}
	});

	
	
	//jQuery.post("/newEnd/purchaseProduct_new.do",{pid:pid,purchaserId:uid,flag:flag},function(data){
		//closeDownLoadDiv('2');
		
		//alert("购买下载  data  = " + data);
		//var index = data.indexOf("1|");
		//if(index == 0){//文档可直接下载
			//var downPath = data.split("|")[1];
			//alert("购买下载  downPath== "+downPath);
			//location.href=downPath;
			//return;
		//}
		
		//downLoadNew(pid,'');
	//});
}
//加载 购买付费文档(弹窗)
function purchasePayProduct(pid,uid){
	var random = new Date().getTime();
	jQuery.ajax({
		type:'GET',
		url:"/newEnd/buyPayProduct_temp.do",
		context: document.body,
		data: "pid="+pid+"&purchaserId="+uid+"&flag=buy&datetime=random",
		success:function(data){
			if(data){
				var oHtml = data;
				jQuery(this).append(oHtml);
				dialogBoxShadow();
				setObjCenter(idflag);
				if(idflag == "downLoadTips1"){
					//下载、购买提示
					if(jQuery(".zk").length>0){
						jQuery(".zk").click(function(){
							var $index=jQuery(".zk").index(this);
							jQuery(".cz_btn").hide();
							jQuery(".cz_btn").eq($index).show();
							jQuery(".zk").show();
							jQuery(this).hide();
						});
					}
				}
		}
		}
	});
}
//关闭下载各种弹窗
function closeDownLoadDiv(id){
		var oDownLoad = jQuery("#downLoadTips"+id);
		if(oDownLoad.length>0){
			oDownLoad.remove();
			dialogBoxHidden();
			if(id == 1){
				var oVipTips = jQuery(".vipShowTips"),
				oDownTip5 = jQuery("#downLoadTips4");
				if(oDownTip5.length>0&&oVipTips.length>0){
					oVipTips.remove();
					oDownTip5.remove();
				}
			}
		}
	}
//以获取豆元继续支付
function showFin(flag){
	closeDownLoadDiv("4");
	if(flag == "down"){//如果购买的是下载
		downLoadNew(tmp_product,'');
	}else{//购买的是付费
		purchasePayProduct(tmp_product,tmp_userId);
	}
}
//老购买弹层获取豆元
function showGetDouyuan(){
	var getDouyuan = jQuery(".get_douyuan"),
	oBtnDy = jQuery("#dy");
	if(getDouyuan.length>0&&oBtnDy.length>0){
		oBtnDy.bind('click',function(){
			getDouyuan.slideDown();
			jQuery(this).hide();
		});
	}
}
//showGetDouyuan();

//顶
function checkTop(flag,url){
	jQuery.ajax({
	url:url,
	success:function(data){
		if(data == 1){//可以顶
			if(flag == 1){//顶
				if(jQuery("#showlab").length>0){
					var preNum = parseInt(jQuery("#showlab").html());
					jQuery("#showlab").html(preNum+1);
				}
				if(jQuery("#topNum").length>0){
					var preNum = parseInt(jQuery("#topNum").html());
					jQuery("#topNum").html(preNum+1);
					  var left = parseInt(jQuery("#topNum").offset().left)+10,
					  top = 10,
					  addP = jQuery('<div id="zhan"><b>+1<\/b></\div>').appendTo(jQuery(".docin_reader_tools"));
					  jQuery('#zhan').css({'position':'absolute','z-index':'1', 'color':'#2284e2','left':left+'px','top':top+'px'}).animate({top:top-10,left:left+10},'slow',function(){addP.fadeIn('fast').remove();});


				}
			}
			else if(flag == 2){
				if(jQuery("#steponlab").length>0){
					var preNum = parseInt(jQuery("#steponlab").html());
					jQuery("#steponlab").html(preNum+1);

					 var left = parseInt(jQuery("#top_hit").offset().left)+10,
					  top = 10,
					  addP = jQuery('<div id="zhan"><b>+1<\/b></\div>').appendTo(jQuery(".docin_reader_tools"));
					  jQuery('#zhan').css({'position':'absolute','z-index':'1', 'color':'#2284e2','left':left+'px','top':top+'px'}).animate({top:top-10,left:left+10},'slow',function(){addP.fadeIn('fast').remove();});
				}
			}
		}
		else if(data == 0){//顶过了
			alert('你已经投过票了，请不要重复投票');
		}
	}
});
}
function setTop(productId, auditLevel) {
	/**
	 * 1 取出cookie_id的值
	 * 2 判断key = productId的这个cookie是否存在
	 * 3 比较两个值是否相等
	 */
	if (auditLevel != 1 && auditLevel != 20) {
		return;
	}
	var productIdValue = getCookie2008_1(productId);
	var cookieIdValue = getCookie2008_1("cookie_id");
	if (cookieIdValue == null) {
		alert("\u4f60\u7684\u6d4f\u89c8\u5668\u4e0d\u652f\u6301cookie\u529f\u80fd,\u8bf7\u786e\u8ba4!");
		return;
	}
	var url = "/jsp_cn/jquery/end/update_message.jsp?pid=" + productId + "&flag=top_new&date=" + new Date().getTime();
	if (productIdValue == null || productIdValue == "") {	//没有顶过
		setCookie2008_1(productId, cookieIdValue, 60 * 24 * 365);
		checkTop(1,url);
		//jQuery("#a5").load(url);
	} else {
		if (productIdValue != cookieIdValue) {	//该用户没有顶过
			setCookie2008_1(productId, cookieIdValue, 60 * 24 * 365);
			checkTop(1,url);
		} else {
			alert("\u4f60\u5df2\u7ecf\u6295\u8fc7\u7968\u4e86\uff0c\u8bf7\u4e0d\u8981\u91cd\u590d\u6295\u7968");
		}
	}
	shareEvent(productId, 14);
}
//踩
function setStepon(productId, auditLevel) {
	if (auditLevel != 1 && auditLevel != 20) {
		return;
	}
	var productIdValue = getCookie2008_1(productId);
	var cookieIdValue = getCookie2008_1("cookie_id");
	if (cookieIdValue == null) {
		alert("\u4f60\u7684\u6d4f\u89c8\u5668\u4e0d\u652f\u6301cookie\u529f\u80fd,\u8bf7\u786e\u8ba4!");
		return;
	}
	var url = "/jsp_cn/jquery/end/update_message.jsp?pid=" + productId + "&flag=stepon_new&date=" + new Date().getTime();
	if (productIdValue == null) {	//没有顶过
		setCookie2008_1(productId, cookieIdValue, 60 * 24 * 365);
		checkTop(2,url);
		//jQuery("#a5").load(url);
	} else {
		if (productIdValue != cookieIdValue) {	//该用户没有顶过
			setCookie2008_1(productId, cookieIdValue, 60 * 24 * 365);
			checkTop(2,url);
			//jQuery("#a5").load(url);
		} else {
			alert("\u4f60\u5df2\u7ecf\u6295\u8fc7\u7968\u4e86\uff0c\u8bf7\u4e0d\u8981\u91cd\u590d\u6295\u7968");
		}
	}
}


/**********************登陆后调用 付费阅读 ************************/

//用户登录后付费阅读
function clickPayRead(pid,uid){
	//先判断是否登陆
	if(isUserLogin == 0){
		showPayReadLogin("payReadValue");
	}else{
		//加载付费阅读 弹窗
		purchasePayProduct(pid,uid);
	}
	
}

function showPayReadLogin(id){
	if(typeof(id) == 'undefined' || id == null)
		return;
	
	var cook = new CookieClass();
	cook.expires = 10;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("payReadClickId", id);
	
	changeCookieValue(cook,"payReadClickId");
}

function showFeekLogin(){
	
	var cook = new CookieClass();
	cook.expires = 10;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("showFeekClickId", "showfeek");
	
	changeCookieValue(cook,"showFeekClickId");
	
}

//初始化付费阅读cookie   
function initPayRead(){
	
	var cook = new CookieClass();
	var remindClickId = cook.getCookie("payReadClickId");
	if(remindClickId == "payReadValue" && tmp_userId > 0){
		if(iscanread == 0){
			purchasePayProduct(tmp_product,tmp_userId);
		}
	}
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("payReadClickId", -1);
}
/***********************************************/
//将不是cookie_id的cookie值变为-1
function changeCookieValue(cook,cookie_id){
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	
	if(cookie_id != "remindClickId"){
		//将豆单的cookie变为-1
		cook.setCookie("remindClickId", -1);
	}
	
	if(cookie_id != "booksaveClickId"){
		//收藏
		cook.setCookie("booksaveClickId", -1);
	}
	
	if(cookie_id != "downloadClickId"){
		//下载
		cook.setCookie("downloadClickId", -1);
	}
	
	if(cookie_id != "payReadClickId"){
		//付费阅读
		cook.setCookie("payReadClickId", -1);
	}
	
	if(cookie_id != "showFeekClickId"){
		//举报
		cook.setCookie("showFeekClickId", -1);
	}
	
	if(cookie_id != "vip_alert_adv"){
		//相关文档 关闭广告
		cook.setCookie("vip_alert_adv", -1);
	}
	
	if(cookie_id != "can_copy_alert"){
		//播放器复制
		cook.setCookie("can_copy_alert", -1);
	}
	
	showlogin();
	
}


/*****************  登陆后调用 下载   ************/
function showDownloadLogin(id){
	if(typeof(id) == 'undefined' || id == null)
		return;
	var cook = new CookieClass();
	cook.expires = 10;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("downloadClickId", id);
	
	changeCookieValue(cook,"downloadClickId");
}

//用户登录后下载
function clickDownload(pid,fn){
	//先判断是否登陆
	if(isUserLogin == 0){
		showDownloadLogin("downloadValue");
	}else{
		downLoadNew(pid,fn);
	}
	
}
//初始化下载cookie   
function initdownload(){

	var cook = new CookieClass();
	var remindClickId = cook.getCookie("downloadClickId");
	if(remindClickId == "downloadValue" && tmp_userId > 0){
		downLoadNew(tmp_product,isFreeDoc);
	}
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("downloadClickId", -1);
}

/***************   结束   **************/

/***************** 终极页 登陆后 调用 收藏***********************/
function showBookSaveLogin(id){
	if(typeof(id) == 'undefined' || id == null)
		return;
	var cook = new CookieClass();
	cook.expires = 10;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("booksaveClickId", id);
	
	changeCookieValue(cook,"booksaveClickId");
}

//用户登录后收藏
function clickBookSave(pid){
	//先判断是否登陆
	if(isUserLogin == 0){
		showBookSaveLogin("booksaveValue");
	}else{
		addDocToBookshop(pid);
	}
	
}
//初始化收藏cookie   
function initbooksave(){

	var cook = new CookieClass();
	var remindClickId = cook.getCookie("booksaveClickId");
	if(remindClickId == "booksaveValue" && tmp_userId > 0){
		addDocToBookshop(tmp_product);
	}
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("booksaveClickId", -1);
}


/***************** 结束 ***********************/




//用户登录后加入豆单
function initInsertDoclist(pid){

	var cook = new CookieClass();
	var remindClickId = cook.getCookie("remindClickId");
	if(remindClickId == "insertDoclist" && tmp_userId > 0){
		insertDoclist(pid);
	}
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("remindClickId", -1);
}

//终极页加入豆单按钮
function insertDoclist(pid) {
	
	if(isZuZhi == "1"){
		mailDialogAlert(login_email,landingEmail,domailUrl)
		return;
	}
	
	//请求载入提示层
	var doudanUrl = "/jsp_cn/productEnd/ajax/doclist_pop_new.jsp?random=" + Math.round(Math.random() * 100);
	if (jQuery("#doudanRelative").length == 0 && tmp_userId > 0) {
		jQuery.ajax({type:"POST", url:doudanUrl, data:"productid=" + pid, async:false, success:function (re) {
			if (re) {
				var popTmp = jQuery("<div id=\"doudanRelative\"></div>");
				popTmp.html(re);
				dialogBoxShadow();
				popTmp.appendTo(document.body);
			}
		}});
	}
	//发送ajax请求  判断用户是否创建过豆单
	var url = "/my/docin/doclist/insertdoclist.do?random=" + Math.round(Math.random() * 100);
	jQuery.post(url, null, function (data) {
		if (data != null) {
			var json = eval("(" + data + ")");
			var result = json.result;
			if (result == 0) {
				//未登录 弹出登陆框
				showLoginAndClick("insertDoclist");
			} else {
				if (result == 1) {
				//已登录 没有豆单
					var oCreateDiv = document.getElementById("createDoudan");
					setAlertCenter(oCreateDiv);
				} else {
					if (result == 2) {
				//已登录 有豆单
						var doclist = json.doclist;
						if (doclist != null) {
							document.getElementById("doclistEndUL").innerHTML = "";
							for (var dl in doclist) {
								var id = doclist[dl].id;
								var title = doclist[dl].title;
								if (title != undefined) {
									var n = document.createElement("li");
									n.innerHTML = "<input type='radio' name='doclistEndId' value='" + id + "' /><label>" + title + "</label>";
									document.getElementById("doclistEndUL").appendChild(n);
								}
							}
							var n2 = document.createElement("li");
							document.getElementById("doclistEndUL").appendChild(n2);
							var n1 = document.createElement("li");
							var url = "/app/my/docin/doclist/createdoclist";
							if (document.getElementById("productId") != null) {
								var productId = document.getElementById("productId").value;
								url = "/app/my/docin/doclist/createdoclist?pid=" + productId;
							}
							n1.style.fontSize = 14 + "px";
							n1.innerHTML = "\u76ee\u524d\u6ca1\u6709\u9002\u5408\u7684\u8c46\u5355,<a target='_blank' onclick='displayNoneAddDoudan();' href='" + url + "'>\u521b\u5efa\u8c46\u5355</a>";
							document.getElementById("doclistEndUL").appendChild(n1);
							var oAddDiv = document.getElementById("addDoudan");
							setAlertCenter(oAddDiv);
						}
					}
				}
			}
		}
	});
}
function displayNoneAddDoudan(){

	document.getElementById('addDoudan').style.display='none';

}

function addProduct2Doclist(){

	var productId = document.getElementById("doclistProductId").value;

    var doclistEnd = document.getElementsByName("doclistEndId");

    var doclistId ;

    var i = 0;

    for (i=0; i<doclistEnd.length; i++)

    {

        if (doclistEnd.item(i).checked)

        {

            doclistId = doclistEnd.item(i).getAttribute("value");

            break;

        }

    }
    closeDialogDiv('addDoudan',1);

	var url = "/my/docin/doclist/adddoclistend.do?random="+Math.round(Math.random()*100)+"&productid="+productId+"&doclistid="+doclistId;

	jQuery.post(url,null,function(data){

		if(data != null){

			if(data == 1){
				dialogBoxShadow();
				var oAddOkDiv=document.getElementById('addOk');

				setAlertCenter(oAddOkDiv);

				//2秒后 关闭弹出层

				setTimeout( function test(){closeDialogDiv('addOk',1);},2000);

			}

		}

	});

}

//弹出层绝对居中定位
function setAlertCenter(obj) {
	if (!obj) {
		return false;
	}
	var d = document;
	obj.style.display = "block";
	var data = {ow:obj.clientWidth, oh:obj.clientHeight, vw:(function () {
		if (d.compatMode == "BackCompat") {
			return d.body.clientWidth;
		} else {
			return d.documentElement.clientWidth;
		}
	})(), vh:(function () {
		if (d.compatMode == "BackCompat") {
			return d.body.clientHeight;
		} else {
			return d.documentElement.clientHeight;
		}
	})(), st:(d.body.scrollTop || d.documentElement.scrollTop)};
	obj.style.left = (data.vw - data.ow) / 2 + "px";
	obj.style.margin = 0;
	obj.style.position = "absolute";
	obj.style.top = (data.vh - data.oh) / 2 + data.st + "px";
}
//举报
function showFeekback(obj, elem) {
	var d = document, o = d.getElementById(obj), s = d.getElementById(elem);
	if (!o || !s) {
		return false;
	}
	s.onclick = function () {
		var sObj = new getXYWH(this);
		_left = (sObj.showL > 800) ? ("right:68px;") : ("left:" + (sObj.showL) + "px;");
		o.style.cssText = ";top:" + (sObj.showT + 25) + "px;" + _left;
		if (elem == "recommendMessageButtun") {
			var productId = d.getElementById("productId").value;
			var sendUrl = "/app/jquery/isRecommendProduct?pid=" + productId + "&rand=" + new Date().getTime();
			var result = trim(jQuery.ajax({url:sendUrl, async:false}).responseText);
			alert(result);
			if (result == "true") {
				d.getElementById("feekBackDiv").style.display = "block";
				d.getElementById("recommendedMessageDiv").style.display = "block";
				d.getElementById("reportMessageDiv").style.display = "none";
			} else {
				d.getElementById("recommendedMessageDiv").style.display = "none";
				d.getElementById("reportMessageDiv").style.display = "none";
				jQuery("#jquerydiv").load("/app/jquery/mingrenrecom?pid=" + productId + "&date=" + new Date().getTime());
			}
		} else {
			d.getElementById("feekBackDiv").style.display = "block";
			d.getElementById("recommendedMessageDiv").style.display = "none";
			d.getElementById("reportMessageDiv").style.display = "block";
		}
	};
}
function hiddenFeekback() {
	var d = document;
	// d.getElementById("subTip").innerHTML = "\u8f93\u5165\u4e0e\u4e4b\u91cd\u590d\u7684\u53e6\u4e00\u7bc7\u6587\u6863URL\u5730\u5740";
	// d.getElementById("subTip").style.color = "#000";
	// d.getElementById("purl").value = "";
	d.getElementById("feekBackDiv").style.display = "none";
}

//发送举报消息（新）
function feekSubmit(id){
	
	var doc = document;
	var content = doc.getElementById("error3").value;//举报内容
	var type = 2;//举报类型
	
	if(doc.getElementById("error6").checked){// 举报该文档为侵权文档。
		type = type + 11;
		content == "" ? doc.getElementById("error6").value : content;
	}else if(doc.getElementById("error1").checked){// 举报该文档含有违规或不良信息。
		type = type + 3;
		content == "" ? doc.getElementById("error1").value : content;
	}else if(doc.getElementById("error4").checked){// 反馈该文档无法正常浏览。
		type = type + 4;
		content == "" ? doc.getElementById("error4").value : content;
	}else if(doc.getElementById("error2").checked){// 举报该文档为重复文档。
		type = type + 5;
		content = doc.getElementById("purl").value;//输入的url
		
		if (content == "" || content.length < 1) {//输入url内容为空
			showFeekError("请输入与本篇重复的文档网址");
			
			return;
		}else{//输入url不为空
			var begin = content.indexOf("-");
			var end = content.lastIndexOf(".html");
			if (begin == -1 || end == -1) {//输入格式不正确
				showFeekError("请输入正确的网址<br />如：http://www.docin.com/p-23020082.html");
				
				return;
			}else{
				
				var reg = new RegExp("^[0-9]*$");
				var jbid = content.substring(begin + 1, end);
				if(!reg.test(jbid)){
					showFeekError("请输入正确的网址<br />如：http://www.docin.com/p-23020082.html");
					
					return;
				}else if (parseInt(jbid) > parseInt(id)) {//输入的文档上传时间在举报文档上传时间之后 
					showFeekError("您的举报无效，请核实举报文档的上传时间");
					
					return;
				} else if (parseInt(jbid) == parseInt(id)) {//输入文档与举报文档为同一文档
					showFeekError("输入文档链接有误，不能举报同一篇文档");
					
					return;
				}
			}
		}
	}else{
		alert("请您选择举报类型！");
		
		return;
	}
	
	var url = "/jsp_cn/jquery/end/update_message.jsp?content=" + encodeURI(content) + "&pid=" + id + "&type=" + type + "&flag=sendFeekBack&date=" + new Date().getTime();
	jQuery("#a5").load(url);
	
}

//显示举报错误信息
function showFeekError(msg){
	if(jQuery("#feekTips").length>0){
		jQuery("#feekTips").html(msg);
		jQuery("#feekTips").show();
	}else{
		jQuery('<p id="feekTips" class="fcr">'+msg+'</p>').insertAfter(jQuery("#repeatproduct"));
	}
}


//发送举报消息
function sendFeekBack(id) {
	var d = document, content = "", type = 2, errs = "";
	var error3 = d.getElementById("error3").value;
	if (error3 == "") {	//没有填写举报内容testarea
		if (d.getElementById("error1").checked) {
			type = type + 3;
			content = d.getElementById("error1").value;
		} else if(d.getElementById("error6").checked) {//举报该文档为侵权文档
			type = type + 11;
			content = d.getElementById("error6").value;
		}else{
			if (d.getElementById("error2").checked) {
				type = type + 5;
//			content = d.getElementById("error2").value;
				content = d.getElementById("purl").value;
				if (content == "" || content.length < 1) {
					if(jQuery("#feekTips").length>0){
						jQuery("#feekTips").html("请输入与本篇重复的文档网址");
						jQuery("#feekTips").show();
					}
					else{
						jQuery('<p id="feekTips" class="fcr">请输入与本篇重复的文档网址</p>').insertAfter(jQuery("#repeatproduct"));
					}
					return;
				} else {
					var begin = content.indexOf("-");
					var end = content.lastIndexOf(".html");
					if (begin == -1 || end == -1) {
						if(jQuery("#feekTips").length>0){
							jQuery("#feekTips").html("请输入正确的网址<br />如：http://www.docin.com/p-***.html");
							jQuery("#feekTips").show();
						}
						else{
							jQuery('<p id="feekTips" class="fcr">请输入正确的网址<br />如：http://www.docin.com/p-***.html</p>').insertAfter(jQuery("#repeatproduct"));
						}
						return;
					} else {
						var jbid = content.substring(begin + 1, end);
						if (parseInt(jbid) > parseInt(id)) {
							if(jQuery("#feekTips").length>0){
								jQuery("#feekTips").html("您的举报无效，请核实举报文档的上传时间");
								jQuery("#feekTips").show();
							}
							else{
								jQuery('<p id="feekTips" class="fcr">您的举报无效，请核实举报文档的上传时间</p>').insertAfter(jQuery("#repeatproduct"));
							}
							//alert("\u60a8\u7684\u4e3e\u62a5\u65e0\u6548\uff0c\u8bf7\u6838\u5b9e\u4e3e\u62a5\u6587\u6863\u7684\u4e0a\u4f20\u65f6\u95f4");
							return;
						} else {
							if (parseInt(jbid) == parseInt(id)) {
								if(jQuery("#feekTips").length>0){
									jQuery("#feekTips").html("输入文档链接有误，不能举报同一篇文档");
									jQuery("#feekTips").show();
								}
								else{
									jQuery('<p id="feekTips" class="fcr">输入文档链接有误，不能举报同一篇文档</p>').insertAfter(jQuery("#repeatproduct"));
								}
								return;
							}
						}
					}
				}
			} else {
				if (d.getElementById("error4").checked) {
					type = type + 4;
					content = d.getElementById("error4").value;
				} else {
					if (d.getElementById("error5").checked) {
						type = type + 2;
						content = d.getElementById("error5").value;
						var mes = document.getElementsByName("errors");
						for (var i = 0; i < mes.length; i++) {
							if (mes[i].checked) {
								content = content + mes[i] + " ";
							}
						}
					} else {
						alert("\u8bf7\u6dfb\u5199\u6216\u9009\u62e9\u4e3e\u62a5\u4fe1\u606f!");
						return;
					}
				}
			}
		}
	} else {
		content = error3;
		if (d.getElementById("error2").checked) {
			type = type + 5;
//			content = d.getElementById("error2").value;
			content = d.getElementById("error3").value;
			if (content == "" || content.length < 1 || content == "\u8f93\u5165\u4e0e\u4e4b\u91cd\u590d\u7684\u53e6\u4e00\u7bc7\u6587\u6863URL\u5730\u5740") {
				alert("\u8bf7\u8f93\u5165\u91cd\u590d\u6587\u6863!");
				return;
			} else {
				var begin = content.indexOf("-");
				var end = content.lastIndexOf(".html");
				if (begin == -1 || end == -1) {
					alert("\u8f93\u5165\u6b63\u786e\u7f51\u5740");
					return;
				} else {
					var jbid = content.substring(begin + 1, end);
					if (parseInt(jbid) > parseInt(id)) {
						alert("\u60a8\u7684\u4e3e\u62a5\u65e0\u6548\uff0c\u8bf7\u6838\u5b9e\u4e3e\u62a5\u6587\u6863\u7684\u4e0a\u4f20\u65f6\u95f4");
						return;
					} else {
						if (parseInt(jbid) == parseInt(id)) {
							d.getElementById("subTip").innerHTML = "\u8f93\u5165\u6587\u6863\u94fe\u63a5\u6709\u8bef\uff0c\u4e0d\u80fd\u4e3e\u62a5\u540c\u4e00\u7bc7\u6587\u6863\uff01";
							d.getElementById("subTip").style.color = "#f00";
							return;
						}
					}
				}
			}
		}
	}
	var url = "/jsp_cn/jquery/end/update_message.jsp?content=" + encodeURI(content) + "&pid=" + id + "&type=" + type + "&flag=sendFeekBack&date=" + new Date().getTime();
	jQuery("#a5").load(url);
}
//举报弹层
function changeRadio() {
	var d = document;
	var obj = jQuery("#error2");
	if(obj.length>0){
		jQuery("#purl").val("");
		if(obj.prop("checked")){
			jQuery("#repeatproduct").show();
		}
		else{
			if(jQuery('#feekTips').length>0){
				jQuery("#feekTips").hide();
			}
			jQuery("#repeatproduct").hide();
		}
	}
}

function showFeek(){
 	var d = document, o = d.getElementById("feekBackDiv"), s = d.getElementById("feekback1");
	if (!o || !s) {
		return false;
	}
	var sObj = new getXYWH(s);
		_left = (sObj.showL > 800) ? ("right:68px;") : ("left:" + (sObj.showL) + "px;");
		o.style.cssText = ";top:" + (sObj.showT + 25) + "px;" + _left;
	d.getElementById("feekBackDiv").style.display = "block";
	d.getElementById("recommendedMessageDiv").style.display = "none";
	d.getElementById("reportMessageDiv").style.display = "block";
	
}

function isShowFeek(){
	
	if(isUserLogin == 0){//用户未登录
		
		showFeekLogin();
		
	}else{
		showFeek();
	}
	
}

/***************************登录后弹出举报单层*****************************/

//初始化举报cookie   
function initshowfeek(){

	var cook = new CookieClass();
	var remindClickId = cook.getCookie("showFeekClickId");
	if(remindClickId == "showfeek" && tmp_userId > 0){
		showFeek();
	}
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("showFeekClickId", -1);
}

/***************************登录后弹出举报单层*****************************/
//终极页二维码用  start  -----
function showDimCodeDiv(window){
	var feekDimcCode = document.getElementById("feekDimcCode"),
		dimCodeDiv = document.getElementById("dimCodeDiv"),
		docinIcon = document.getElementById("dimcode_docin_icon"),
		dimPic = document.getElementById("dimcode_pic");
		if(!feekDimcCode||!dimCodeDiv){return;}
		feekDimcCode.onclick = function(){
			var l = getDistance(this)[0];
				var t = getDistance(this)[1];
				dimCodeDiv.style.left = l+'px';
				dimCodeDiv.style.top = t+28+'px';
				var doc_url = window.location.href;
				//处理分节阅读url
				var indexof_html=doc_url.indexOf(".html", 0);
		 		var end=doc_url.substring(indexof_html, doc_url.length);
		 		var tmp=doc_url.substring(0,indexof_html);
		 		var start=tmp.substring(0,tmp.indexOf("_",0)>0?tmp.indexOf("_",0):tmp.length);
		 		start=start.substring(0, start.indexOf("-f",0)>0?start.indexOf("-f",0):start.length)
		 		doc_url=start+end;
				dimPic.src = "/servlet/get2wm?doc_end_url=" + doc_url;
				dimCodeDiv.style.display = "block";
				inpmv(4457);
		};
		dimPic.onload = function(){
			docinIcon.style.display = "block";
		}
		
	function closeDimDiv(){
		dimCodeDiv.style.display = "none";
	}
	window.closeDimDiv = closeDimDiv;	
}
(function(){
	showDimCodeDiv(window);//显示二维码
	//showGetDouyuan();//获取豆元渠道
})();
function getDistance(obj){
		var left = 0,
		top = 0;
		while(obj.offsetParent){
			left += obj.offsetLeft;
			top += obj.offsetTop;
			obj = obj.offsetParent;
		}
		return [left,top];
}
//获取豆元
function showGetDouyuan(){
	var getDouyuan = jQuery(".get_douyuan"),
	oBtnDy = jQuery("#dy");
	if(getDouyuan.length>0&&oBtnDy.length>0){
		oBtnDy.bind('click',function(){
			getDouyuan.slideDown();
			jQuery(this).hide();
			jQuery("#dh").hide();
		});
	}
}
//复制功能
function docinCopyData(handlerId,infoDataId){	
	var clip = "clip"+handlerId;
	clip = new ZeroClipboard.Client();
  	clip.setHandCursor(true);  	
  	clip.addEventListener('mouseOver', function (client) {
  		var oText = jQuery('#'+infoDataId);
    	clip.setText(oText.val());
		oText.select();
  	});  	
  	clip.addEventListener('complete', function (client, text) {
    	alert("内容已经复制，你可以使用Ctrl+V 粘贴。");
  	});
  	clip.glue(handlerId);
}
function openDocinshare(n){//参数n  1：人人 2：QQ空间 3：新浪微博 4：腾讯微博 5：QQ好友 6：开心网
	if(!bdshareconfig){return;}
	switch (n){
		case 1:
			bdshareconfig.to = "renren";
			renrenBdShare(js_pid,js_cid);
			break;
		case 2:
			bdshareconfig.to = "qzone";
			qqZoneBdShare(js_pid,js_cid);
			break;
		case 3:
			bdshareconfig.to = "tsina";
			sinaWbBdShare(js_pid,js_cid);
			break;
		case 4:
			bdshareconfig.to = "tqq";
			qqWbBdShare(js_pid,js_cid);
			break;
		case 5:
			bdshareconfig.to = "sqq";
			qqFdBdShare(js_pid,js_cid);
			break;
		case 6:
			bdshareconfig.to = "kaixin001";
			kaixinBdShare(js_pid,js_cid);
			break;
		default:
			return;
	}
	var shareUrl = 'http://s.share.baidu.com/?click=1&url='+bdshareconfig.url+'&uid=&to='+bdshareconfig.to+'&type=text&pic='+bdshareconfig.pic+'&title='+bdshareconfig.title+' &key=&sign=on&desc='+bdshareconfig.desc+'&comment='+bdshareconfig.desc+'&searchPic=1&l=181b8ivcl181b8j0c5181b8jelv&linkid=hk1wfrmedvq&sloc=674.301.0.28.35.20.28.36.37.0.961&apiType=0&buttonType=0&firstime=1372954734986'
	window.open(shareUrl);
}


function closeDialogDiv(id,f){
	jQuery("#"+id).remove();
	if(id == "docinPopBox1"){
		jQuery("#docUrl_copyclick").remove();
		jQuery("#flashUrl_copyclick").remove();
		jQuery("#embedTxt_copyclick").remove();
	}
	if(f){
		dialogBoxHidden();
	}
}
function closeOpenShare(id){
	jQuery("#"+id).hide();
	dialogBoxHidden();
	jQuery("#docUrl_copyclick").remove();
	jQuery("#flashUrl_copyclick").remove();
	jQuery("#embedTxt_copyclick").remove();
}
function changeSize(url,w,h,obj){
var str = '<embed src="'+url+'" width="'+w+'" height="'+h+'" type="application/x-shockwave-flash" ALLOWFULLSCREEN="true" ALLOWSCRIPTACCESS="always"></embed>';
	jQuery("#endShareCode").val(str);
	var aLI=jQuery('#b_size a');
	if(aLI.length>0){
		aLI.removeClass('cur');
		jQuery(obj).addClass('cur');
	}		
} 
function showReportDiv(){
	dialogBoxShadow();
	setObjAbsoluteCenter('feekBackDiv');
}

function gotoCreateDoudan(){
	var oDoudanRelativeDiv = jQuery("#doudanRelative");
	if(oDoudanRelativeDiv.length>0){
		oDoudanRelativeDiv.remove();
		dialogBoxHidden();
	}
}

function startTime(id,h , m , s , from){
	var _t = null;
	clearTimeout(_t);
	var _v = '距离免费结束还有：';
	if(from == 'index_after'){
    	_v = "免费结束：";
    }
    if(from == 'index' && id == 'oTime'){
    	_v += "<br />";
    } 
    _v += h + '小时' + m + '分' + s + '秒';
	var oTimebox = document.getElementById(id);
	if(oTimebox){
		 oTimebox.innerHTML = _v; 
	}
	_t = setTimeout('doTime(\''+id+'\','+  h +' , '+ m +' , '+ s +' , \'' + from + '\')', 1000); 
}

//aliyun fangqueue -- 
function alivisit(productid){
	if(productid != null){
		jQuery.post("/app/aliv/aliv.do", {p:productid}, null);
	}
}
function alivisit_2014(productid){
	if(productid != null){
		jQuery.post("/app/aliv/aliv_2014.do", {p:productid}, null);
	}
}
//aliyun fangqueue -- 

function showMessageContent(){
	var cpage = jQuery("#msgcpage").val(),
		pid = jQuery("#productId").val(),
		total = jQuery("#total").val(),
		userIp = jQuery("#userIp").val(),
		isMy = jQuery("#ismy").val();
		if(isMy == ""){isMy = 0;}
		var url = "/jsp_cn/jquery/end/message_new.jsp?page="+cpage+"&pid="+pid+"&ism="+isMy+"&to="+total+"&userIp="+userIp+"&date="+new Date().getTime();
		jQuery("#showMsgContent").load(url);
}
function showMessageU(currentPage){
	var cpage = currentPage,
	pid = jQuery("#productId").val(),
	total = jQuery("#total").val(),
	isMy = jQuery("#ismy").val();
	if(isMy == ""){isMy = 0;}
	var url = "/jsp_cn/jquery/end/message_new.jsp?page="+cpage+"&pid="+pid+"&ism="+isMy+"&to="+total+"&date="+new Date().getTime();
	jQuery("#showMsgContent").load(url);
}
//留言
function insertUserMessage(notLogin) {
	var d = document,
		messageText = d.getElementById("message"),
		message = trim(messageText.value),
		productId = d.getElementById("productId").value,
		userIp = d.getElementById("userIp").value,
		insertMB = d.getElementById("insertMessageBtn");
	if(message == ""){
		alert("留言文字不能为空！");
		//jQuery("#msgCheck").html('留言文字不能为空！').show();
		hiddenTextareaImp();
		messageText.select();
		return ;
	}
	if(message.length > 1000){
		alert("留言文字太长，不能超过1000个字！");
		//jQuery("#msgCheck").html('留言文字太长，不能超过1000个字！').show();
		messageText.select();
		return ;
	}
	var code ='';
	if(notLogin == 1){//检查验证码
		var oCheckCode = jQuery("#checkout");
		if(oCheckCode.length>0){
			code = oCheckCode.val();
			if(code == ''){
				refCode('regimg');
				alert("验证码为空!");
				//jQuery("#msgCheck").html('验证码为空').show();	
				return ;
			}
		}
	}
	insertMB.disabled = true;
	
	//jQuery("#msgCheck").hide().html('');
	jQuery("#a5").load("/jsp_cn/jquery/end/product_message.jsp",{"pid":productId,"islogin":notLogin,"code":code,"userIp":userIp,"message":encodeURI(message),"date":new Date().getTime()},function(data){
		//alert("hehe");
		refreshFatherWindow();

	});
}
function resetText(txt,div){
   document.getElementById(txt).value = '';
   document.getElementById(div).style.display = 'none';
}

function refreshFatherWindow(){
//	if(window.opener && window.opener != null && !window.opener.closed){
	if (typeof(refresh)!='undefined'){
		if(refresh == 1){
			window.opener.location.href = "http://shequ.docin.com/task/todayTask.do?task_id=110";
		}
	}
	
//	}
}
//隐藏消息提示框
function hiddenTextareaImp(){
	var o = document.getElementById("textarea_imp");
	o.style.display="none";
}
function showTextareaImp(){
	var o = document.getElementById("textarea_imp");
	o.style.display="";
}
//删除消息
function deleteMessage(messageId){	
	var d=document,productId=d.getElementById("productId").value;

	if(confirm("确定要删除吗？")){
		var url = "/jsp_cn/jquery/end/update_message.jsp?pid="+productId+"&mId="+messageId+"&flag=del_mess&date="+new Date().getTime();
		jQuery("#a5").load(url);
	}
}

//隐藏消息标签
function hiddenMessage(messageId){
	 var d=document,o= d.getElementById("userMessageDIV"+messageId);
	 o.style.display="none";
}

//显示回复消息输入框
function showOrHiddenReMessageText(reMsgDivId){	
	var d=document,t=d.getElementById(reMsgDivId); 
	t.style.display=(t.style.display=="")?"none":"";
	//重置滚动条
	if(typeof(frame)!='undefined'){
		frame.sly('reload');
	}
	if(t.style.display==""){
		v='regimg_re'+reMsgDivId.replace("reMsgDiv","");
		refCode(v);
	}
}

//重置验证码
function refCode(img_id){
	var d=document,checkImg=d.getElementById(img_id);
	//alert(checkImg);
	if(checkImg!=null){
		checkImg.src="/servlet/getctime?"+getTime();
	}
}

function getTime(){
	var d, s = "";
	var c = "";
	d = new Date();
	s += d.getYear()+c;
	s += (d.getMonth() + 1) + c;
	s += d.getDate() + c;
	s += d.getHours() + c;
	s += d.getMinutes() + c;
	s += d.getSeconds() + c;
	s += d.getMilliseconds();
	return s;
}

//插入回复信息
function insertReMessage(reMessageTextId,messageId,senderId,notLogin){
	var d=document,
  reMessageText=d.getElementById(reMessageTextId),
  reMessage = reMessageText.value,
  productId=d.getElementById("productId").value,
  userIp = d.getElementById("userIp").value;
	reMessage=trim(reMessage);
	if(reMessage==""){
		alert("留言文字不能为空！");
		reMessageText.select();
		return ;
	}
	if(reMessage.length>1000){
		alert("留言文字太长，不能超过1000个字！");
		reMessageText.select();
		return ;
	}
	var code ='';
	if(notLogin==1){//检查验证码
		code = d.getElementById("checkout_re"+messageId).value;
		if(code==''){
			alert("验证码为空");	
			return ;
		}
	}
	showApply(reMessageTextId);
	jQuery("#a5").load("/jsp_cn/jquery/end/product_message.jsp",{"pid":productId,"islogin":notLogin,"code":code,"userIp":userIp,"message":encodeURI(reMessage),"date":new Date().getTime(),"returnid":messageId});
}

//显示验证码
function showCheckCode(){
	var cv = document.getElementById("checkCodeDiv");
	if(cv!=null&&cv.style.display == "none"){
		cv.style.display="";
		refCode('regimg');
	}
}

//回复消息内容
function setMsgContent(productId){
	var d=document,msg_c_name="msg_content_cookie_"+productId;
	temp=getCookie2008_1(msg_c_name);
	if(temp!=null && temp!="null"){
		d.getElementById("message").value=temp;
	}
}

//显示回复div
function showReMessageDiv(reUserProductMessageId,reMessage){
    var d=document,
     reMsgDiv=	d.getElementById("reMessageContent"+reUserProductMessageId);
     reMsg=reMsgDiv.innerHTML;
     reMsg=reMsg.replace("reMessageCount",reMessage);
     reMsgDiv.innerHTML=reMsg;
     reMsgDiv.style.display="";
}

//设置回复连接href
function setReMsgDelA_href(reMessageId,messageId){
	var d=document,
  a=d.getElementById("reMsgDelA"+messageId);
	if(a!=null){
			a.href="javascript:deleteReMessageAjax("+reMessageId+","+messageId+")";		
	}
}

//删除回复信息
function deleteReMessageAjax(reUserProductMessageId,userProductMessageId){
	var d=document,
  productId=d.getElementById("productId").value;
//reMessageContent
	if(confirm("确定要删除吗？")){
		var url = "/jsp_cn/jquery/end/update_message.jsp?reUserProductMessageId="+reUserProductMessageId+
			"&pid="+productId+"&userProductMessageId="+userProductMessageId+"&flag=deleteReMessage&date="+new Date().getTime();
		jQuery("#a5").load(url);
	}
}

//显示回复连接
function hiddenReMsgA(userProductMessageId){	   
		var d=document;
  d.getElementById("reMsgA"+userProductMessageId).style.display="none";
}

//删除回复信息
function deleteReMessage(reUserProductMessageId,userProductMessageId){
	var d=document,
  productId=d.getElementById("productId").value;
//reMessageContent
	if(confirm("确定要删除吗？")){
		var url = "/jsp_cn/jquery/end/update_message.jsp?reUserProductMessageId="+reUserProductMessageId+
			"&pid="+productId+"&userProductMessageId="+userProductMessageId+"&flag=deleteReMessage&date="+new Date().getTime();
		jQuery("#a5").load(url);
	}
}

//隐藏回复div
function hiddenReMessageDiv(reUserProductMessageId){
	 var d=document,
   reMsgDiv=	d.getElementById("reMessageContent"+reUserProductMessageId);
  reMsgDiv.style.display="none";
}

//显示回复连接
function showReMsgA(userProductMessageId){	   
		var d=document;
  d.getElementById("reMsgA"+userProductMessageId).style.display="";
}
//不刷新页面加载消息
function ajaxAddUserMsg(){
	var d=document,
  userMessageDiv = d.createElement("ajaxMessageTemplet");
	msg = d.getElementById("ajaxMessageTemplet").innerHTML;
	msg1 = d.getElementById("message").value;
	msg1 = msg1.replace(/</ig,"&lt;");
	msg1 = msg1.replace(/\r\n/ig,"<br/>");
	msg = msg.replace("userMessage",msg1);
	var messageCount = Math.floor(d.getElementById("messageCount").value)+1;
	d.getElementById("messageCount").value = messageCount;
	msg = msg.replace("messageCount",messageCount);
	//msg = msg.replace("messageCount","");
	userMessageDiv.innerHTML = msg;
	var addMsgDiv = d.getElementById("ajaxAddUserMessage");
	var showMsgDiv = d.getElementById("ajaxAddUserMessageBottom");
	showMsgDiv.innerHTML = addMsgDiv.innerHTML+showMsgDiv.innerHTML;
	addMsgDiv.innerHTML = "";
	addMsgDiv.innerHTML = msg;
	var total = d.getElementById("total").value;
	if(total == 0){
		var con = d.getElementById("showMsgContent").innerHTML;
		con = con.replace("暂无评论。","");
		d.getElementById("showMsgContent").innerHTML = con;
	}
	var t = parseInt(total) + parseInt(1);
	d.getElementById("total").value = t;
}
function showApply(id){
	var pid = id.substring(9);
	var oUserTx = jQuery("#tempAply .userSpeak").html();
	var msg = jQuery('#'+id).val();
	var oTempAply = jQuery('<div class="listPl listHf"><div class="cont contF"><p>'+oUserTx+'</p><p>'+msg+'</p></div></div>');
	oTempAply.appendTo(jQuery('#msgContent'+pid));
}
//前帖片跳过广告
function showJumpAdv(){
	if(if_show_jump){
		showlogin_jump();
	}else{

		jQuery.ajax({
			type:'GET',
			url:"/newEnd/jumpAdv_new.do",
			context: document.body,
			data: "product_id="+tmp_product,
			success:function(data){
				if(data){
					var oHtml = data;
					jQuery(this).append(oHtml);
					var oPenVip = jQuery("#docVipOpen");
					if(oPenVip.length>0){
						dialogBoxShadow();
						setObjCenter('docVipOpen');
						var $aInput=jQuery("input[name='vip']");
						$aInput.each(function(){
							if(jQuery(this).is(":checked")){
								jQuery(this).parent("label").css("color","#669900");
							}
						});
						$aInput.click(function(){
							$aInput.parent("label").css("color","#333");
							jQuery(this).parent("label").css("color","#669900");
						});
					}
				}
			}
		});
	}
}

function showlogin_jump(){
		jQuery("#loginwindow").load("/jsp_cn/weibo/login.jsp",function (data){
			var key = "show_jump_div";
			var value = "1";
			var exp = new Date();
			exp.setTime(exp.getTime() + 15*1000);
			var jump_cookie = key + "="+ escape (value) + ";expires=" + exp.toGMTString();
			document.cookie = jump_cookie;
			dialogBoxShadow();
			if(document.getElementById("newlogin")){
				setObjCenter("newlogin");
				jQuery('input[placeholder]').placeholder();
				jQuery('input[placeholder]').bind("focus",function(){
					jQuery(this).addClass("txtFocus");
				});
				jQuery('input[placeholder]').bind("blur",function(){
					jQuery(this).removeClass("txtFocus");
				});
			}
			document.getElementById("username_new").focus();
		});
}
//轻松获取豆元
function showGetDouyuan(){
	var getDouyuan = jQuery(".get_douyuan"),
	oBtnDy = jQuery("#dy");
	if(getDouyuan.length>0&&oBtnDy.length>0){
		oBtnDy.bind('click',function(){
			getDouyuan.slideDown();
			jQuery(this).hide();
		});
	}
}


function changeAlert(){
	closeDownLoadDiv("1");
	dialogBoxShadow();
	setObjCenter("downLoadTips4");
}
//函数功能：json 排序  
// filed:(string)排序字段，  
// reverse: (bool) 是否倒置(是，为倒序)  
// primer (parse)转换类型 
function sortBy(filed, reverse, primer){
	reverse = (reverse) ? -1 : 1;
	return function (a, b) {  
		a = a[filed];  
		b = b[filed];  
		  
		if (typeof (primer) != "undefined") {  
		a = primer(a);  
		b = primer(b);  
		} 
		if(reverse == 1){
			return a-b;
		}
		else{
			return b-a;
		}
	}
}
//调用  
function sortCourseTime_ASC(clickType) {
	if(typeof(dd) == "undefined" || dd.length == 0 || jQuery("#addFavSuccess").length == 0){
		jQuery('<p style="text-align:center;">暂无数据!</p>').appendTo(jQuery(".userBehavMod .bd"));
		return;
	}
	if(clickType == 1){//按收藏排序
		dd.sort(sortBy('adv', true, parseInt));
		var str = "收藏该文档的用户也收藏",
		pmv = 5510;
	}
	else if(clickType == 2){//按下载排序
		dd.sort(sortBy('down', true, parseInt));
		var str = "下载该文档的用户也下载",pmv = 5511;
	}
	if(jQuery(".userBehavMod .bd").length>0){return false;}
	var oDocWarp = jQuery('<div class="inner"><div class="hd"><h3>'+str+'</h3></div><div class="bd"></div>').appendTo(jQuery(".userBehavMod"));
	var oDocListUl = jQuery('<ul class="doc_list clear"></ul>').appendTo(jQuery(".userBehavMod .bd"));
	for(var i = 0;i< dd.length;i++){
		if(i == 4){
			break;
		}
		var oLiItem = jQuery('<li><div class="li_wrap"><div class="cover"><a href="/p-'+dd[i].pid+'.html" onmousedown="return inpmv('+pmv+');" target="_blank"><img src="http://img1.vonibo.com/docin_'+dd[i].pid+'_110x140.jpg" alt="'+dd[i].title+'"><span class="pageno">'+dd[i].pageno+'</span></a></div><p class="title"><a href="/p-'+dd[i].pid+'.html" onmousedown="return inpmv(5319);" target="_blank" title="'+dd[i].title+'">'+dd[i].title+'</a></p></div></li>').appendTo(oDocListUl);
	}
}

//一下是下载进度条js
function getHttpRequest(){
	var xmlHttp;
	try{
		//firefox,opera 8.0+,safari 浏览器
		xmlHttp=new XMLHttpRequest();
	}catch(e){
		try{
			//ie
			xmlHttp=new ActiveXobject("Msxml2.XMLHTTP");
		}catch(e){
			try{
				xmlHttp=new ActiveXobject("Microsoft.XMLHTTP");
			}catch(e){
				alert("得不到xmlhttp");
			}
		}
	}
	return xmlHttp;
}

function download(url,su,productId){
	var ajax=getHttpRequest();
	ajax.open("GET", url, true);
	ajax.responseType = "blob";
	jQuery('.down_btns').append('<div class="outerProg"><div class="downProg"><div class="progBar"></div><div class="progNum progNumA"><b></b></div><div class="progNum progNumB"><b></b></div></div></div>');
	ajax.onreadystatechange = function () {
        if (ajax.readyState == 4) {
            if (ajax.status==200){
            	su(ajax.response);
            	downloadSuccess(productId);
            } 
        }
    };
	//设置进度监听事件，实时监听进度
	ajax.addEventListener("progress", function(e){
		var currLoaded = e.loaded;
		var percent = Math.floor((currLoaded / e.total) * 100) + "%";

		jQuery('.progNumA').children('b').text(percent);
		jQuery('.progNumB').children('b').text(percent);
		jQuery('.progBar').width(Math.floor((currLoaded / e.total) * 102) + "%");
		jQuery('.progNumB').width(percent);	
		
		if(e.loaded == e.total){
			jQuery('.outerProg').remove();
		}
	});
	ajax.send(null);
}

function DocinReaderPlayer(){
	if(typeof(isBuildImagePlayer) == "undefined"){return false;};
	if(arguments.length==1){ 
		var arg = arguments[0];
	}
	this.config = {
		productId:0,
		allPage:0,
		previewNum:0,
		isBuy:1,
		baseUrl:"",
		showStyle:1,
		contStyle:1,
		bookMark:0,
		isLogin:0,
		bookmarkid:-1,
		showAd:true,
		yjAdvStyle:'00',
		load_url:""
	};

	for(var p in arg){
		this.config[p] = arg[p];
	}
	this.iWidth = [792,1130];
	this.iHeight = [];
	this.metas={0:{width:720,height:540},1:{width:720,height:540},2:{width:720,height:540}};//记录文档高度
	this.screenWidth = window.screen.width;
	this.btn_comment = jQuery(".page_comment");
	this.btn_upload = jQuery(".page_upload");
	this.menu = jQuery('.doc_readmod,.reader_menu');
	this.btn_search = jQuery(".page_search");
	this.topBtn = jQuery(".top_like");
	this.hitBtn = jQuery(".step_on");
	this.zoomFlag = 0;
	this.adTj = {'1':[923504,923505,923506,923507,923508],'2':[939345,939347,939348,939349,939350]};
	this.init(); 
}
DocinReaderPlayer.prototype = {
	init:function(){
		var _this = this;
		_this.readerMenu();
		_this.initComment();
		//分享
		_this.btn_upload.bind('click',function(){
			_this.openShare(_this.config.productId);
		});
		if(jQuery(".new_share_more").length>0){
			jQuery(".new_share_more").bind("click",function(){
				_this.openShare(_this.config.productId);
			});
		}
		//付费阅读购买
		var oPayBtn = jQuery("#payForRead"),
			oGotoReadBtn = jQuery("#gotoRead");
		if(oGotoReadBtn.length>0){
			oGotoReadBtn.bind('click',function(){
				purchaseProduct(js_pid,tmp_userId,'buy');
			});
		}
		if(oPayBtn.length>0){
			oPayBtn.bind('click',function(){
				//if(_this.config.isLogin == 0){
					//showlogin();
					//return false;
				//}
				//purchasePayProduct(_this.config.productId,tmp_userId);//余额不足，弹出充值层
				clickPayRead(_this.config.productId,tmp_userId);
			});
		}
		jQuery("#bookmarktips .close").bind('click',function(event){
			jQuery("#bookmarktips").fadeOut();
			event.stopPropagation();
		});
	},
	readerMenu:function(){
		var _this = this;
		var menuList = jQuery(".menu_list_wrap");
		var otime = null;
		if(menuList.length<0){return false;}
		_this.menu.bind('mouseover',function(){
			clearTimeout(otime);
			menuList.show();
		}).bind('mouseleave',function(){
			otime = setTimeout(function(){
				menuList.hide();
			},100);
			
		});
		//菜单功能
		var oShowInfo = jQuery("#j_showinfo"),
		oShowAbout = jQuery("#j_about"),
		oJlist = jQuery("#j_list"),
		oJpic = jQuery("#j_pic");
		if(oShowInfo.length>0){
			oShowInfo.bind('click',function(){
				var docInfo = new CreateDocinDialog(showInfoConfig);
			});
		}
		if(oShowAbout.length>0){
			oShowAbout.bind('click',function(){
				var docAbout = new CreateDocinDialog({
					id:'docAbout',
					title:'关于豆丁',
					button:2,
					shadow:1,
					content:'<span class="fwb">豆丁网是中国最领先的C2C文档分享网站。</span><br/>豆丁网(www.docin.com)创立于2008年，致力于打造全球最大的中文社会化阅读平台，为用户提供一切有价值的可阅读之物。经过几年的运营，豆丁网已经成功跻身互联网全球500强，成为提供垂直服务的优秀网站之一。目前豆丁网拥有分类广泛的实用文档、众多出版物、行业研究报告、以及数千位行业名人贡献的专业文件，各类读物总数超过两亿，是目前全球最大的中文文档库。'
				});
			});
		}
	}, 
	commentHandel:function(){//jpg终极页评论功能
		var _this = this;
		jQuery.ajax({
			url:'/jsp_cn/productEnd/load/doc_end_load_comment.jsp',
			data:'isLogin='+_this.config.isLogin,
			context: document.body,
			success:function(data){
				var oHtml = data;
				jQuery(this).append(oHtml);
				var cH = jQuery(window).height();
				var oCommentHeight = cH - cH*0.2 -142;
				jQuery("#showMsgContent").height(oCommentHeight);
				dialogBoxShadow();
				setObjCenter('j_comment');
				showMessageContent();
				var oTextArea = jQuery("#message");
				oYzm = jQuery("#checkout");
				if(oTextArea.length == 0){return;}
				oTextArea.bind('keydown',function(event){
					if(event.ctrlKey&&event.keyCode == 13){
						insertUserMessage(1);
					}
				});
				if(oYzm.length>0){
					oYzm.bind('keydown',function(event){
						if(event.ctrlKey&&event.keyCode == 13){
							insertUserMessage(1);
						}
					});
				}
				oTextArea.bind('focus',function(){
					showCheckCode();
				});
				jQuery('textarea[placeholder]').placeholder();
				oTextArea.focus();
			}
			});
	},
	initComment:function(){
		var _this = this;
		jQuery(".page_comment,#showComm").bind('click',function(){
			_this.commentHandel();
		});
	},
	openShare:function(pid){//打开分享页面
		var popTmp = jQuery("#docinPopBox1");
			if(popTmp.length>0){
				dialogBoxShadow();
				setObjCenter('docinPopBox1');
				docinCopyData('docUrl','textUrl');
				docinCopyData('flashUrl','flUrl');
				docinCopyData('embedTxt','endShareCode');
			}
			/**shareEvent(pid,13);
			//add at 2013-04-25 分享弹窗请求
			jQuery.ajax({
				type:"POST",
				url:"/jsp_cn/productEnd/share_new_v1.jsp",
				context: document.body,
				data:"productid="+pid,
				async: false,
				success:function(re){
					if(re){
						jQuery(this).append(re);
						dialogBoxShadow();
						setObjCenter('docinPopBox1');
						docinCopyData('docUrl','textUrl');
						docinCopyData('flashUrl','flUrl');
						docinCopyData('embedTxt','endShareCode');
					}
				}
			});**/
	}
}
var docinReader = new DocinReaderPlayer(readerConfig);
showGetDouyuan();
initInsertDoclist(product_id);
initdownload();
initbooksave();
initPayRead();
initshowfeek();