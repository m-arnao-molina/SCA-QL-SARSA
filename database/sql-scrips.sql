TRUNCATE TABLE datos_ejecucion CASCADE;
TRUNCATE TABLE datos_iteracion CASCADE;
TRUNCATE TABLE resultado_ejecucion CASCADE;

ALTER SEQUENCE datos_ejecucion_seq RESTART WITH 1;
ALTER SEQUENCE datos_iteracion_id_seq RESTART WITH 1;
ALTER SEQUENCE resultado_ejecucion_id_seq RESTART with 1;

with resumen as (
    SELECT de.nombre_algoritmo,
           COUNT(distinct de.id)                                        as ejecuciones_indicadas,
           min(de.inicio)                                               as PRIMER_INICIO,
           max(de.inicio)                                               as ULT_INICIO,
           max(de.fin)                                                  as ULT_TERMINO,
           SUM(CASE WHEN de.estado in ('pendiente') THEN 1 else 0 END)  AS PENDIENTES,
           SUM(CASE WHEN de.estado in ('ejecutando') THEN 1 else 0 END) AS EJECUTANDO,
           SUM(CASE WHEN de.estado in ('terminado') THEN 1 else 0 END)  AS TERMINADO

    FROM datos_ejecucion de
    WHERE nombre_algoritmo in ('SCAQL_SCP', 'GWOQL_SCP', 'SCA_SCP', 'GWO_SCP')
    group by de.nombre_algoritmo
)

select * from resumen;


/* limpieza pendientes */

DELETE FROM datos_iteracion where id_ejecucion in (

        SELECT id
        FROM datos_ejecucion
        WHERE estado in ('ejecutando')
          and nombre_algoritmo in ('GWOQL_SCP', 'SCAQL_SCP', 'GWO_SCP', 'SCA_SCP')

    );

UPDATE datos_ejecucion set estado = 'pendiente'
WHERE   estado in ('ejecutando')
      and nombre_algoritmo in ('GWOQL_SCP', 'SCAQL_SCP', 'GWO_SCP', 'SCA_SCP');



/* validaciones */

 SELECT *
        FROM datos_ejecucion
        WHERE estado in ('ejecutando')
          and nombre_algoritmo in ('GWOQL_SCP', 'SCAQL_SCP', 'GWO_SCP', 'SCA_SCP');


SELECT * FROM datos_iteracion

where id in (

    SELECT id
        FROM datos_ejecucion
        WHERE estado in ('ejecutando')
          and nombre_algoritmo in ('GWOQL_SCP', 'SCAQL_SCP', 'GWO_SCP', 'SCA_SCP')
    )
;


/* revision "ejecutando" */


select de.id as id_ejec,
       de.nombre_algoritmo,
       max(di.numero_iteracion) as ult_iter

from datos_ejecucion de
left join datos_iteracion di on de.id = di.id_ejecucion
where   de.nombre_algoritmo in ('SCAQL_SCP', 'GWOQL_SCP', 'SCA_SCP', 'GWO_SCP')
    and de.estado in ('ejecutando')

group by de.id,
         de.nombre_algoritmo
;

/* querys */ 
select de.id,
       count(di.id) as num_iters
from datos_ejecucion de
left join datos_iteracion di on de.id = di.id_ejecucion
where
        de.nombre_algoritmo in ('SCAQL_SCP', 'GWOQL_SCP', 'SCA_SCP', 'GWO_SCP')
    and de.estado in ('terminado')

group by de.id
;


/*CONSULTA PARA CHEQUEAR CUANTAS ITERACIONES SE REALIZO POR ALGORITMO*/ 

select de.id, de.nombre_algoritmo, count(di.id), de.estado from datos_ejecucion de
	left join datos_iteracion di on di.id_ejecucion = de.id 
	where de.nombre_algoritmo in ('SCAQL_SCP', 'GWOQL_SCP', 'SCA_SCP', 'GWO_SCP', 'SCA_SCP_MIR2','GWO_SCP_MIR2') and de.estado = 'terminado'
	group by de.id
	having count(di.id) > 0



/* QUERYS PARA CORRECCION DE EXPERIMENTOS QUE QUEDARON EJECUTANDO */

--1) Consultar el estado actual de los experimentos

with resumen as (
    SELECT de.nombre_algoritmo,
           COUNT(distinct de.id)                                        as ejecuciones_indicadas,
           min(de.inicio)                                               as PRIMER_INICIO,
           max(de.inicio)                                               as ULT_INICIO,
           max(de.fin)                                                  as ULT_TERMINO,
           SUM(CASE WHEN de.estado in ('pendiente') THEN 1 else 0 END)  AS PENDIENTES,
           SUM(CASE WHEN de.estado in ('ejecutando') THEN 1 else 0 END) AS EJECUTANDO,
           SUM(CASE WHEN de.estado in ('terminado') THEN 1 else 0 END)  AS TERMINADO

    FROM datos_ejecucion de
    WHERE nombre_algoritmo in (
    'GWO_SCP_BCL1_CPU_C','SCA_SCP_BCL1_CPU_C','HHO_SCP_BCL1_CPU_C','WOA_SCP_BCL1_CPU_C',
    'GWO_SCP_MIR2_CPU_C','SCA_SCP_MIR2_CPU_C','HHO_SCP_MIR2_CPU_C','WOA_SCP_MIR2_CPU_C',
    'GWO_SCP_QL1_CPU_C','SCA_SCP_QL1_CPU_C','HHO_SCP_QL1_CPU_C','WOA_SCP_QL1_CPU_C',
    'GWO_SCP_QL2_CPU_C','SCA_SCP_QL2_CPU_C','HHO_SCP_QL2_CPU_C','WOA_SCP_QL2_CPU_C',
    'GWO_SCP_QL3_CPU_C','SCA_SCP_QL3_CPU_C','HHO_SCP_QL3_CPU_C','WOA_SCP_QL3_CPU_C',
    'GWO_SCP_QL4_CPU_C','SCA_SCP_QL4_CPU_C','HHO_SCP_QL4_CPU_C','WOA_SCP_QL4_CPU_C',
    'GWO_SCP_QL5_CPU_C','SCA_SCP_QL5_CPU_C','HHO_SCP_QL5_CPU_C','WOA_SCP_QL5_CPU_C'
    )
    group by de.nombre_algoritmo
)

select * from resumen;


--2) Eliminar datos_iteracion de los experimentos que no terminaron

DELETE FROM datos_iteracion WHERE id_ejecucion in (

        SELECT id as id_ejecucion
        FROM datos_ejecucion
        where estado in ('ejecutando')
            and nombre_algoritmo in (
            'GWO_SCP_BCL1_CPU_C','SCA_SCP_BCL1_CPU_C','HHO_SCP_BCL1_CPU_C','WOA_SCP_BCL1_CPU_C',
            'GWO_SCP_MIR2_CPU_C','SCA_SCP_MIR2_CPU_C','HHO_SCP_MIR2_CPU_C','WOA_SCP_MIR2_CPU_C',
            'GWO_SCP_QL1_CPU_C','SCA_SCP_QL1_CPU_C','HHO_SCP_QL1_CPU_C','WOA_SCP_QL1_CPU_C',
            'GWO_SCP_QL2_CPU_C','SCA_SCP_QL2_CPU_C','HHO_SCP_QL2_CPU_C','WOA_SCP_QL2_CPU_C',
            'GWO_SCP_QL3_CPU_C','SCA_SCP_QL3_CPU_C','HHO_SCP_QL3_CPU_C','WOA_SCP_QL3_CPU_C',
            'GWO_SCP_QL4_CPU_C','SCA_SCP_QL4_CPU_C','HHO_SCP_QL4_CPU_C','WOA_SCP_QL4_CPU_C',
            'GWO_SCP_QL5_CPU_C','SCA_SCP_QL5_CPU_C','HHO_SCP_QL5_CPU_C','WOA_SCP_QL5_CPU_C'
            )

);

--3) Eliminar resultado_ejecucion de los experimentos que no terminaron

DELETE FROM resultado_ejecucion WHERE id_ejecucion in (

                SELECT id as id_ejecucion
                FROM datos_ejecucion
                where estado in ('ejecutando')
                    and nombre_algoritmo in (
                    'GWO_SCP_BCL1_CPU_C','SCA_SCP_BCL1_CPU_C','HHO_SCP_BCL1_CPU_C','WOA_SCP_BCL1_CPU_C',
                    'GWO_SCP_MIR2_CPU_C','SCA_SCP_MIR2_CPU_C','HHO_SCP_MIR2_CPU_C','WOA_SCP_MIR2_CPU_C',
                    'GWO_SCP_QL1_CPU_C','SCA_SCP_QL1_CPU_C','HHO_SCP_QL1_CPU_C','WOA_SCP_QL1_CPU_C',
                    'GWO_SCP_QL2_CPU_C','SCA_SCP_QL2_CPU_C','HHO_SCP_QL2_CPU_C','WOA_SCP_QL2_CPU_C',
                    'GWO_SCP_QL3_CPU_C','SCA_SCP_QL3_CPU_C','HHO_SCP_QL3_CPU_C','WOA_SCP_QL3_CPU_C',
                    'GWO_SCP_QL4_CPU_C','SCA_SCP_QL4_CPU_C','HHO_SCP_QL4_CPU_C','WOA_SCP_QL4_CPU_C',
                    'GWO_SCP_QL5_CPU_C','SCA_SCP_QL5_CPU_C','HHO_SCP_QL5_CPU_C','WOA_SCP_QL5_CPU_C'
                    )

);


--4) Actualizar experimentos en "ejecutando" a "pediente"

UPDATE datos_ejecucion
SET estado = 'pendiente'
WHERE id in (

        SELECT id as id_ejecucion
        FROM datos_ejecucion
        where estado in ('ejecutando')
            and nombre_algoritmo in (
            'GWO_SCP_BCL1_CPU_C','SCA_SCP_BCL1_CPU_C','HHO_SCP_BCL1_CPU_C','WOA_SCP_BCL1_CPU_C',
            'GWO_SCP_MIR2_CPU_C','SCA_SCP_MIR2_CPU_C','HHO_SCP_MIR2_CPU_C','WOA_SCP_MIR2_CPU_C',
            'GWO_SCP_QL1_CPU_C','SCA_SCP_QL1_CPU_C','HHO_SCP_QL1_CPU_C','WOA_SCP_QL1_CPU_C',
            'GWO_SCP_QL2_CPU_C','SCA_SCP_QL2_CPU_C','HHO_SCP_QL2_CPU_C','WOA_SCP_QL2_CPU_C',
            'GWO_SCP_QL3_CPU_C','SCA_SCP_QL3_CPU_C','HHO_SCP_QL3_CPU_C','WOA_SCP_QL3_CPU_C',
            'GWO_SCP_QL4_CPU_C','SCA_SCP_QL4_CPU_C','HHO_SCP_QL4_CPU_C','WOA_SCP_QL4_CPU_C',
            'GWO_SCP_QL5_CPU_C','SCA_SCP_QL5_CPU_C','HHO_SCP_QL5_CPU_C','WOA_SCP_QL5_CPU_C'
        )

);


------

SELECT id as id_ejecucion
FROM datos_ejecucion
where estado in ('pendiente')
    and nombre_algoritmo in (
    'GWO_SCP_BCL1_CPU_C','SCA_SCP_BCL1_CPU_C','HHO_SCP_BCL1_CPU_C','WOA_SCP_BCL1_CPU_C',
    'GWO_SCP_MIR2_CPU_C','SCA_SCP_MIR2_CPU_C','HHO_SCP_MIR2_CPU_C','WOA_SCP_MIR2_CPU_C',
    'GWO_SCP_QL1_CPU_C','SCA_SCP_QL1_CPU_C','HHO_SCP_QL1_CPU_C','WOA_SCP_QL1_CPU_C',
    'GWO_SCP_QL2_CPU_C','SCA_SCP_QL2_CPU_C','HHO_SCP_QL2_CPU_C','WOA_SCP_QL2_CPU_C',
    'GWO_SCP_QL3_CPU_C','SCA_SCP_QL3_CPU_C','HHO_SCP_QL3_CPU_C','WOA_SCP_QL3_CPU_C',
    'GWO_SCP_QL4_CPU_C','SCA_SCP_QL4_CPU_C','HHO_SCP_QL4_CPU_C','WOA_SCP_QL4_CPU_C',
    'GWO_SCP_QL5_CPU_C','SCA_SCP_QL5_CPU_C','HHO_SCP_QL5_CPU_C','WOA_SCP_QL5_CPU_C'
    );



