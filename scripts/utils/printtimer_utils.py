import numpy
import sys
from typing import TextIO, Any
import builtins

def formatfloat2str(val:float, width:int=10, ndigits:int=4) -> str:
    """
    Format float into string. Typically used for printing timers.
    If the fixed point format exceeds the specified width,
    scientific exponential notation is used instead
    to ensure string remains within specified width.
    - width: total character length of output string
    - ndigits: number of digits following the decimal point
    """
    # Try fixed point notation, e.g., '   12.3400'
    result = f"{val:{width}.{ndigits}f}"
    if len(result) > width:
        # Try scientific notation if fixed point string is too long, e.g., '1.2340e+01'
        precision = builtins.max(0, width - 6)
        result = f"{val:{width}.{precision}e}"
    return result

def make_header_rows_parallel(
    *,
    lminmax: bool | int,
    ltime: bool | int,
    namewidth: int = 18,
) -> list[str]:
    """Return header lines for printtimers (parallel)"""
    line_table: list[str]
    namepadding = namewidth * ' '
    line_table = [
        namepadding     + '        Total time         Deviation',
        namepadding     + '  (all CPUs)   (per CPU)            ',
        namepadding     + '      (s)         (s)         (s)   ',
    ]
    if lminmax:
        line_table[0] += '      Min         Max'
        line_table[1] += '                     '
        line_table[2] += '      (s)         (s)'
    if ltime:
        line_table[0] += '     Time per step'
        line_table[1] += '       (per CPU)'
        line_table[2] += '          (s)'

    return line_table

def make_header_rows_serial(
    *,
    ltime: bool | int,
    namewidth: int = 19,
) -> list[str]:
    """Return header lines for printtimers (serial)"""
    line_table: list[str]
    namepadding = namewidth * ' '
    line_table = ['', '']
    if not ltime:
        line_table[0] = namepadding      + ' Total time'
        line_table[1] = namepadding      + '       (s)'
    else:
        line_table[0] = namepadding      + ' Total time          Time per step'
        line_table[1] = namepadding      + '       (s)                  (s)'
    return line_table

def write_vlist(
    vlist: numpy.ndarray | Any,
    *,
    ff: TextIO | Any,
    name: str,
    mintime: float,
    lminmax: bool | int,
    top_it: int | None,
):
    from ..warp import me, npes
    def f2s(val):
        return formatfloat2str(val, width=10, ndigits=4)
    if me > 0: return
    if not isinstance(vlist, numpy.ndarray):
        vlist = numpy.array(vlist)
    vsum = vlist.sum()
    if vsum <= mintime: return
    vrms = vlist.std()
    ff.write(f"{name}  {f2s(vsum)}  {f2s(vsum/npes)}  {f2s(vrms)}")
    if lminmax:
        vmin = vlist.min()
        vmax = vlist.max()
        ff.write(f"  {f2s(vmin)}  {f2s(vmax)}")
    if top_it is not None and top_it > 0:
        ff.write(f"   {f2s(vsum/npes/(top_it))}")
    ff.write('\n')
