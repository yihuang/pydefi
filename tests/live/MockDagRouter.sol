// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function balanceOf(address) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
}

interface IUniswapV2Pair {
    function swap(uint amount0Out, uint amount1Out, address to, bytes calldata data) external;
    function token0() external view returns (address);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32);
}

struct BaseRequest {
    uint256 fromToken; address toToken; uint256 fromTokenAmount; uint256 minReturnAmount; uint256 deadLine;
}
struct RouterPath {
    address[] mixAdapters; address[] assetTo; uint256[] rawData; bytes[] extraData; uint256 fromToken;
}

contract UniswapV2Adapter {
    function sellBase(address to, address pool, bytes calldata extraData) external { _swap(to, pool, extraData); }
    function sellQuote(address to, address pool, bytes calldata extraData) external { _swap(to, pool, extraData); }
    function _swap(address to, address pool, bytes calldata extraData) private {
        address tokenIn  = address(bytes20(extraData[0:20]));
        require(extraData.length >= 40, "ed");
        address t0 = IUniswapV2Pair(pool).token0();
        (uint256 r0, uint256 r1) = _getReserves(pool);
        uint256 amtIn = IERC20(tokenIn).balanceOf(pool);
        if (amtIn == 0) return;
        bool isT0In = tokenIn == t0;
        uint256 rIn  = isT0In ? r0 : r1;
        uint256 rOut = isT0In ? r1 : r0;
        uint256 dep  = amtIn > rIn ? amtIn - rIn : 0;
        if (dep == 0) return;
        uint256 out = (dep * 997 * rOut) / (rIn * 1000 + dep * 997);
        if (out == 0) return;
        if (isT0In) IUniswapV2Pair(pool).swap(0, out, to, "");
        else         IUniswapV2Pair(pool).swap(out, 0, to, "");
    }
    function _getReserves(address pool) private view returns (uint256 r0, uint256 r1) {
        (uint112 _r0, uint112 _r1,) = IUniswapV2Pair(pool).getReserves();
        r0 = _r0; r1 = _r1;
    }
}

contract MockDagRouter {
    uint256 constant AM = 0x000000000000000000000000ffffffffffffffffffffffffffffffffffffffff;
    uint256 constant RV = 0x8000000000000000000000000000000000000000000000000000000000000000;
    uint256 constant WM = 0x00000000000000000000ffff0000000000000000000000000000000000000000;
    uint256 constant IM = 0x0000000000000000ff0000000000000000000000000000000000000000000000;
    uint256 constant OM = 0x000000000000000000ff00000000000000000000000000000000000000000000;

    event DagNode(uint256 indexed idx, uint256 edges, uint256 balance, address to);

    function dagSwapTo(uint256, address receiver, BaseRequest calldata req, RouterPath[] calldata paths) external payable returns (uint256) {
        require(paths.length > 0, "empty");
        address ft = address(uint160(req.fromToken & AM));
        uint256 bal = IERC20(ft).balanceOf(address(this));
        require(bal >= req.fromTokenAmount, "insuf");
        _run(receiver, bal, paths);
        uint256 ret = IERC20(req.toToken).balanceOf(receiver);
        require(ret >= req.minReturnAmount, "minRet");
        return ret;
    }

    function _run(address receiver, uint256 bal, RouterPath[] calldata paths) private {
        uint256 total = paths.length;
        for (uint256 i = 0; i < total;) {
            if (i > 0) {
                bal = IERC20(address(uint160(paths[i].fromToken & AM))).balanceOf(address(this));
                require(bal > 0, "zero");
            }
            emit DagNode(i, paths[i].mixAdapters.length, bal, address(this));
            _node(receiver, bal, i, paths[i], total);
            unchecked { ++i; }
        }
    }

    function _node(address receiver, uint256 bal, uint256 idx, RouterPath calldata p, uint256 total) private {
        uint256 n = p.mixAdapters.length;
        require(n > 0, "noE");
        uint256 tw;
        uint256 acc;
        for (uint256 i = 0; i < n;) {
            uint256 rd = p.rawData[i];
            uint256 w  = (rd & WM) >> 160;
            require((rd & IM) >> 184 == idx, "iIdx");
            uint256 oIdx = (rd & OM) >> 176;
            tw += w;
            if (i == n - 1) require(tw == 10000, "total");
            uint256 amt = (i == n - 1) ? bal - acc : (bal * w) / 10000;
            acc += (i == n - 1) ? 0 : amt;
            if (amt > 0) IERC20(address(uint160(p.fromToken & AM))).transfer(p.assetTo[i], amt);
            // Intermediate nodes: output to address(this). Last node's edges: output to receiver.
            address to = (idx == total - 1) ? receiver : address(this);
            _edge(p.rawData[i], p.mixAdapters[i], p.extraData[i], to);
            unchecked { ++i; }
        }
    }

    function _edge(uint256 rd, address adapter, bytes memory extra, address to) private {
        address pool = address(uint160(rd & AM));
        string memory sig = (rd & RV) != 0 ? "sellQuote(address,address,bytes)" : "sellBase(address,address,bytes)";
        (bool ok,) = address(adapter).call(abi.encodeWithSignature(sig, to, pool, extra));
        require(ok, "adapter");
    }
}
