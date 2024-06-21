# Multi Threading Notes

The purpose of threading is to speed up programs(but not necessary, wit-h concurrency!!)

Concurrency : - Running blocks of code, in a particular order one after another





<p> Here we ran the code synchronously, and called the same function multiple times one after.<br>
When the function was sleeping, we waited first for it to finish sleeping then exited and then called the function again.
</p>

<ul>
<li>CPU bound tasks : Tasks that primarily use CPU processing. e.g: - Data Crunching, Arithmetic Operations etc.</li>
<li>I/O bound tasks: - Tasks that primarily use I/O processing. e.g.: - Reading and Writing from File System, I/O Operation, Network operations</li>
</ul>
<br>
Threading are useful only for I/O Operations. Threading can also slow down, due to thread management.


